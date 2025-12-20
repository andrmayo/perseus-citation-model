import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import cast, Iterable, Iterator

import torch
import transformers
from bs4 import BeautifulSoup
from bs4.element import NavigableString, _RawAttributeValues, Tag


import perscit_model.extraction.evaluate as evaluate
from perscit_model.extraction.data_loader import ID2LABEL, ExtractionDataLoader
from perscit_model.extraction.inference import InferenceModel


class CitationTagger:
    cit_elements: Iterable[str] = ("quote", "bibl")

    def __init__(
        self,
        model_path: str | Path,
        window_size: int = 512,
        stride: int | None = None,
        center: int | None = None,
        device: str | None = None,
        **kwargs,
    ):
        """
        Args:
            model_path: path to model to load
            window_size: this should probably be equal to the training context window
            stride: number of tokens to shift context window by, by default half of the window size
            center: defaults to stride, which ensures that every token winds up in center for some window
        """
        self.inference_model = InferenceModel(model_path, **kwargs)
        self.loader: ExtractionDataLoader = self.inference_model.loader
        self.window_size: int = window_size
        # this ensures that every token windw up in the reliable center of context window
        self.stride: int = stride if stride else window_size // 2
        self.center: int = center if center else self.stride
        # model in self.inference_model should have moved itself to GPU if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._batch_size = None

    # The approach here is to preprocess and tokenize the whole XML file
    # and then slide a window through the tokens
    def process_xml(
        self,
        xml_path: str | Path | Iterable[str] | Iterable[Path],
        preserve_existing: bool = True,
        overwrite: bool = True,
    ):
        """
        Args:
            xml_path: path or Iterable of paths
            preserve_existing: if True, do not overwrite existing citation tags
            copy: if True, create a copy of all XML files with tags added rather than modifying in place
        """
        if not isinstance(xml_path, Iterable):
            xml_path = [xml_path]

        # Parallelize file processing
        try:
            with ProcessPoolExecutor() as executor:
                executor.map(
                    self.process_xml_file,
                    xml_path,
                    repeat(preserve_existing),
                    repeat(overwrite),
                )
        except Exception as e:
            raise e

    def process_xml_file(
        self, xml_path: Path, preserve_existing: bool = True, overwrite: bool = True
    ) -> None:
        """
        Function mean to process a single XML file. Loads whole file but streams predictions.

        Args:
            xml_path: Path object giving path to XML file.
            preserve_existing: if True (default), keeps any citation tags in file unaltered, if False, replaces with predictions.

        Returns:
            Generator of characters representing character-level labels.
        """
        xml_content = xml_path.read_text(encoding="utf-8")

        # Preprocessing: handle existing citations if preserve_existing = True, strip citation tags, and then tokenize

        # citations as returned by strip_citation_tags is a bit complex:
        # it's a list of tuples, where the first element
        # of each tuple is another tuple giving a citation span (start, end, tag)
        # and the second element is a string containing all the attributes from citation element
        if preserve_existing:
            xml_content, citations = self.strip_citation_tags(xml_content)
        else:
            # discards all preexisting citation tags in XML
            xml_content, citations = evaluate.strip_xml_tags(xml_content), None
        encoding: transformers.BatchEncoding = self.loader.tokenize_text(xml_content)
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        offset_mapping = encoding.offset_mapping[0]

        # Stream xml strings with citation tags inserted based on predicted labels to a tmp file,
        # then replace original xml with tmp file
        # NOTE: Much of the value of streaming here is lost due to usind a DOM
        # with bs4 below, but in the future that could by swapped out with a SAX parser
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=xml_path.parent, delete=False, suffix=".tmp"
        ) as f:
            temp_path = Path(f.name)
            for (
                batch_strings,
                batch_encodings,
                batch_labels,
            ) in self._stream_labels_batched(input_ids, attention_mask, offset_mapping):
                # Write a batch of characters to XML file with <cit>, <bibl>, and <quote> tags inserted
                xml_string = "".join(
                    self.inference_model.insert_tags_into_xml(
                        batch_strings, batch_encodings, batch_labels, citations
                    )
                )
                f.write(xml_string)

        # this approach to wrapping adjacent <bibl> <quote> pairs in a <cit> tag
        # is somewhat inefficient, but much cleaner than incorporating this into
        # the core XML processing logic
        cit_wrapped_xml = self.post_process(temp_path.read_text())
        temp_path.write_text(cit_wrapped_xml)
        # atomic replace of original xml file
        if overwrite:
            try:
                temp_path.replace(xml_path)
            except Exception as e:
                temp_path.unlink()
                raise e
        else:
            # create dir for processed xml files if overwrite = False
            # done here rather than in orchestration function since xml files
            # may live in different directories
            copy_dir = xml_path.parent / "processed"
            copy_dir.mkdir(exist_ok=True)
            try:
                temp_path.replace(copy_dir / xml_path.name)
            except Exception as e:
                temp_path.unlink()
                raise e

    @property
    def batch_size(self) -> int:
        """Quick heuristic for batch size based on available hardware."""
        if self._batch_size:
            return self._batch_size
        if self.device == "cpu":
            self._batch_size = 4
            return self._batch_size

        if self.device == "cuda":
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
            except Exception as e:
                raise e
            # VRAM based heuristic
            if gpu_memory_gb > 24:
                self._batch_size = 256
            if gpu_memory_gb >= 16:
                self._batch_size = 128
            elif gpu_memory_gb >= 8:
                self._batch_size = 64
            else:
                self._batch_size = 32
        else:
            self._batch_size = 4

        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int) -> None:
        print(f"Setting batch size to {new_batch_size}")
        self._batch_size = new_batch_size

    def _stream_labels_batched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: torch.Tensor | list[tuple[int, int]],
        batch_size: int | None = None,
    ) -> Iterator[tuple[str, transformers.BatchEncoding, list[str]]]:
        """
        Args:
            input_ids: BatchEncoding attribute with token ids
            attention_mask: batch_encoding attribute
            offset_mapping: Tensor where offset_mapping[i] gives a Tensor of shape (seq_length, 2)
            batch_size: dynamically set by default
        Returns:
            Iterator over tuples with the arguments for InferenceModel.insert_tags_into_xml
        """
        if batch_size is None:
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size
        # Collect windows into batches
        windows = []
        window_indices = []

        # Create sliding windows
        n_tokens = len(input_ids)
        for window_start in range(0, n_tokens, self.stride):
            window_end = min(window_start + self.window_size, n_tokens)
            # Get window tokens
            window_tokens = (
                input_ids[window_start:window_end].unsqueeze(0).to(self.device)
            )
            window_mask = (
                attention_mask[window_start:window_end].unsqueeze(0).to(self.device)
            )
            window_offsets = offset_mapping[window_start:window_end]
            if isinstance(window_offsets, torch.Tensor):
                window_offsets = window_offsets.to(self.device)
            windows.append(
                {
                    "input_ids": window_tokens,
                    "attention_mask": window_mask,
                    "offset_mapping": window_offsets,
                }
            )
            window_indices.append((window_start, window_end))

            # When batch is full, process it
            if len(windows) == batch_size:
                yield from self._process_window_batch(windows, window_indices)
                windows = []
                window_indices = []

        # get last (incomplete) batch
        if windows:
            yield from self._process_window_batch(windows, window_indices)

    def _process_window_batch(
        self,
        windows: list[dict[str, torch.Tensor]],
        window_indices: list[tuple[int, int]],
    ) -> Iterator[tuple[str, transformers.BatchEncoding, list[str]]]:
        max_len = max(w["input_ids"].shape[0] for w in windows)

        batch_input_ids = torch.stack(
            [
                torch.nn.functional.pad(
                    w["attention_mask"], (0, max_len - len(w["input_ids"]))
                )
                for w in windows
            ]
        )
        batch_attention_mask = torch.stack(
            [
                torch.nn.functional.pad(
                    w["attention_mask"], (0, max_len - len(w["attention_mask"]))
                )
                for w in windows
            ]
        )
        batch_offsets = torch.stack(
            [
                torch.nn.functional.pad(
                    w["offset_mapping"], (0, max_len - len(w["offset_mapping"]))
                )
                for w in windows
            ]
        )

        # Run batched inference
        batch_input_ids = batch_input_ids.to(self.device)
        batch_attention_mask = batch_attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.inference_model.model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )
        predictions = outputs.logits.argmax(dim=-1)
        for i, (start, end) in enumerate(window_indices):
            window_length = end - start
            labels = [ID2LABEL[p] for p in predictions[i, :window_length].tolist()]
            yield self._get_center(
                labels,
                start,
                end,
                batch_input_ids[i],
                batch_attention_mask[i],
                batch_offsets[i],
            )

    @classmethod
    def _handle_cit_elt(cls, soup: BeautifulSoup, tag: Tag) -> None:
        # skip if already inside a cit tag
        if tag.find_parent("cit"):
            return

        next_sibling = tag.next_sibling

        # If next sibling is whitespace-only, get the tag afterwards
        next_tag = None
        whitespace_between = None

        if next_sibling and isinstance(next_sibling, NavigableString):
            # check if it's only whitespace
            as_str = str(next_sibling)
            if (not as_str) or as_str.isspace():
                whitespace_between = next_sibling
                next_tag = next_sibling.next_sibling
            else:
                return
        elif next_sibling and isinstance(next_sibling, Tag):
            next_tag = next_sibling
        else:
            return

        # Check if next tag is complementary type for <cit>
        if (
            next_tag
            and isinstance(next_tag, Tag)
            and next_tag.name in cls.cit_elements
            and tag.name != next_tag.name
        ):
            cit = soup.new_tag("cit")
            tag.insert_before(cit)
            cit.append(tag.extract())
            if whitespace_between:
                cit.append(whitespace_between.extract())
            cit.append(next_tag.extract())

        return

    @staticmethod
    def _handle_orphans(soup: BeautifulSoup, tag: Tag) -> None:
        next_sibling = tag.next_sibling
        if not isinstance(next_sibling, Tag) or tag.name != next_sibling.name:
            return

        merged_tag = soup.new_tag(tag.name, attrs=cast(_RawAttributeValues, tag.attrs))
        tag.insert_before(merged_tag)

        merged_tag.extend(tag.contents)
        # Remove the original tag
        tag.extract()

        # Keep merging as long as next tag is the same
        while isinstance(next_sibling, Tag) and next_sibling.name == merged_tag.name:
            temp_next = next_sibling.next_sibling
            merged_tag.extend(next_sibling.contents)
            next_sibling.extract()
            next_sibling = temp_next
        return

    @classmethod
    def post_process(cls, xml_string: str) -> str:
        """
        This handles citation tags that have been split
        across windows, and also encloses neighboring
        <bibl> and <quote> tags in a <cit> tag
        """
        soup = BeautifulSoup(xml_string, "lxml")

        for tag in soup.find_all(cls.cit_elements):
            cls._handle_orphans(soup, tag)
        for tag in soup.find_all(cls.cit_elements):
            cls._handle_cit_elt(soup, tag)

        return str(soup.body.decode_contents()) if soup.body else str(soup)

    def _get_center(
        self,
        labels: list[str],
        start: int,
        end: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: torch.Tensor | list[tuple[int, int]],
    ) -> tuple[str, transformers.BatchEncoding, list[str]]:
        """
        Args:
            labels: the labels for the tokens in a given window
            start: start index for the window
            end: end index for the window
            offset_mapping: Tensor where offset_mapping[i] gives a Tensor of shape (seq_length, 2)
        Note:
            Offset_mapping is a Tensor if tokenization has return_tensors="pt", by default it's a list of tuples.
            Since the whole file is tokenized at once, there's only one sequence here.
        """
        # Determine reliable center region of window
        if end - start < self.window_size:
            reliable_start, reliable_end = start, end
        else:
            if start == 0:
                reliable_start = 0
                reliable_end = self.stride + (self.window_size - self.stride) // 2
            else:
                reliable_start = start + (self.window_size - self.stride) // 2
                reliable_end = reliable_start + self.stride

        center_ids = input_ids[start:end]
        center_text = self.loader.tokenizer.decode(center_ids)
        center_encoding = transformers.BatchEncoding(
            {
                "input_ids": center_ids,
                "attention_mask": attention_mask[start:end],
                "offset_mapping": offset_mapping[start:end],
            }
        )
        return (
            center_text,
            center_encoding,
            [labels[i] for i in range(reliable_start, reliable_end)],
        )

    #
    # 4. Merge predictions (character-level)
    #
    # 5. Filter conflicts with exisitng citations
    #
    # 6. Wrap bibl-quote pairs
    #
    # 7. Insert tags

    def strip_citation_tags(
        self, xml_content: str
    ) -> tuple[str, list[tuple[int, int, str, str]]]:
        """
        Strip citation tags and track their positions in the stripped text.

        Returns:
            (stripped_text, citations)
            where citations is a list of (start_char, end_char, tag_type)
            and positions are character offsets in stripped_text
        """

        # Parse XML with BeautifulSoup
        soup = BeautifulSoup(xml_content, "lxml")

        # Build stripped text while tracking citation positions
        citations = []
        stripped_parts = []
        current_pos = 0

        def traverse(element: Tag) -> None:
            """Recursively traverse the parse tree, building stripped text."""
            nonlocal current_pos

            # Check if this element is a citation tag
            is_citation = element.name.lower() in (
                "bibl",
                "quote",
                "cit",
            )

            start_pos = None
            if is_citation:
                # Record the start position for citation
                start_pos = current_pos
            else:
                # For non-citation tags, include the opening tag with attributes
                attrs = "".join(
                    f' {k}="{v}"' if isinstance(v, str) else f' {k}="{" ".join(v)}"'
                    for k, v in element.attrs.items()
                )
                opening_tag = f"<{element.name}{attrs}>"
                stripped_parts.append(opening_tag)
                current_pos += len(opening_tag)

            # Process the element's contents
            for child in element.children:
                if isinstance(child, NavigableString):
                    # Add text content to stripped output
                    text = str(child)
                    stripped_parts.append(text)
                    current_pos += len(text)
                elif isinstance(child, Tag):
                    # Recursively process child tags
                    traverse(child)

            if is_citation:
                # Record the citation span
                end_pos = current_pos
                cit_attrs = "".join(
                    f' {k}="{v}"' if isinstance(v, str) else f' {k}="{" ".join(v)}"'
                    for k, v in element.attrs.items()
                )
                citations.append((start_pos, end_pos, element.name.upper(), cit_attrs))
            else:
                # For non-citation tags, include the closing tag
                closing_tag = f"</{element.name}>"
                stripped_parts.append(closing_tag)
                current_pos += len(closing_tag)

        # Start traversal from the root
        # lxml parser wraps content in <html><body>, so process body's children
        body = soup.find("body")
        if body:
            for child in body.children:
                if isinstance(child, NavigableString):
                    text = str(child)
                    stripped_parts.append(text)
                    current_pos += len(text)
                elif isinstance(child, Tag):
                    traverse(child)
        else:
            # Fallback in case no body is found
            traverse(soup)
        stripped_text = "".join(stripped_parts)

        # outer tags come before inner tags
        citations.sort(key=lambda x: (x[0], -x[1]))

        return stripped_text, citations
        # return stripped_text, citations
