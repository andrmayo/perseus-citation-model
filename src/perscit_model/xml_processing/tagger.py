import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Iterable, Iterator

from lxml import etree
from lxml.etree import _Element
import torch
import transformers


import perscit_model.extraction.evaluate as evaluate
from perscit_model.extraction.data_loader import ID2LABEL, ExtractionDataLoader
from perscit_model.extraction.inference import InferenceModel
from perscit_model.shared.xml_utils import get_opening_tag, get_attrs_as_string

logger = logging.getLogger(__name__)


class CitationTagger:
    cit_elements: Iterable[str] = ("quote", "bibl")

    @staticmethod
    def _get_tag_name(element: _Element) -> str:
        """Safely get tag name from lxml element, handling QNames and other types."""
        try:
            return etree.QName(element).localname
        except (ValueError, TypeError):
            # Fallback for non-standard tag types
            return str(element.tag) if hasattr(element, "tag") else ""

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
        encoding: transformers.BatchEncoding = self.loader.tokenize_text(
            xml_content, return_offsets_mapping=True
        )
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
        cit_wrapped_xml = self.post_process(temp_path.read_text(), xml_path)
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
        max_len = max(w["input_ids"].shape[1] for w in windows)

        batch_input_ids = torch.stack(
            [
                torch.nn.functional.pad(
                    w["input_ids"].squeeze(0), (0, max_len - w["input_ids"].shape[1])
                )
                for w in windows
            ]
        )
        batch_attention_mask = torch.stack(
            [
                torch.nn.functional.pad(
                    w["attention_mask"].squeeze(0),
                    (0, max_len - w["attention_mask"].shape[1]),
                )
                for w in windows
            ]
        )
        batch_offsets = torch.stack(
            [
                torch.nn.functional.pad(
                    w["offset_mapping"],
                    (0, 0, 0, max_len - w["offset_mapping"].shape[0]),
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
    def _handle_cit_elt(cls, element: _Element) -> None:
        """Wrap adjacent <bibl> and <quote> pairs in a <cit> tag."""
        # Skip if already inside a cit tag
        for ancestor in element.iterancestors():
            if cls._get_tag_name(ancestor).lower() == "cit":
                return

        parent = element.getparent()
        if parent is None:
            return

        # Find element's index in parent's children
        try:
            index = list(parent).index(element)
        except ValueError:
            return

        # Check for next sibling
        if index + 1 >= len(parent):
            return

        next_elem = parent[index + 1]

        # Check if element's tail (text after element) is whitespace-only
        has_non_whitespace_between = element.tail and not element.tail.isspace()

        # If there's non-whitespace text between, don't wrap
        if has_non_whitespace_between:
            return

        # Check if next element is complementary type for <cit>
        if (
            cls._get_tag_name(next_elem).lower() in cls.cit_elements
            and cls._get_tag_name(element).lower()
            != cls._get_tag_name(next_elem).lower()
        ):
            # Create new cit element
            cit = etree.Element("cit")

            # Save tails before moving elements
            element_tail = (
                element.tail
            )  # Space between bibl and quote (stays inside cit)
            next_elem_tail = next_elem.tail  # Text after quote (becomes cit's tail)

            # Insert cit at current element's position
            parent.insert(index, cit)

            # Move current element into cit
            parent.remove(element)
            cit.append(element)
            # Keep element's tail - it's now the space between bibl and quote inside cit
            element.tail = element_tail

            # Move next element into cit
            parent.remove(next_elem)
            cit.append(next_elem)
            # Clear next_elem's tail since it's now inside cit
            next_elem.tail = None

            # The tail from next_elem becomes cit's tail (text after </cit>)
            cit.tail = next_elem_tail

        return

    @staticmethod
    def _handle_orphans(root: _Element, element: _Element) -> None:
        """Merge adjacent citation elements of the same type (e.g., two <bibl> tags)."""
        parent = element.getparent()
        if parent is None:
            return

        # Find the element's index in parent's children
        try:
            index = list(parent).index(element)
        except ValueError:
            return

        # Check if next sibling has the same tag
        if index + 1 < len(parent):
            next_elem = parent[index + 1]
            # Only merge if they have the same tag AND are truly adjacent
            # (no text or only whitespace between them)
            has_non_whitespace_between = element.tail and not element.tail.isspace()
            if (
                CitationTagger._get_tag_name(next_elem)
                == CitationTagger._get_tag_name(element)
                and not has_non_whitespace_between
            ):
                # Merge next element into current element
                # Concatenate text and children
                if element.text:
                    if not len(element):  # No children
                        # Just append next element's content to this element's text
                        if next_elem.text:
                            element.text = element.text + next_elem.text
                    else:
                        # Has children - append to last child's tail
                        last_child = element[-1]
                        if last_child.tail:
                            last_child.tail = last_child.tail + (next_elem.text or "")
                        else:
                            last_child.tail = next_elem.text
                else:
                    element.text = next_elem.text

                # Move all children from next element to current element
                for child in list(next_elem):
                    element.append(child)

                # Preserve the tail of the next element
                if next_elem.tail:
                    element.tail = (element.tail or "") + next_elem.tail
                else:
                    element.tail = next_elem.tail

                # Remove the next element
                parent.remove(next_elem)

                # Recursively check if there are more siblings to merge
                CitationTagger._handle_orphans(root, element)
        return

    @classmethod
    def post_process(cls, xml_string: str, xml_path: Path | None = None) -> str:
        """
        This handles citation tags that have been split
        across windows, and also encloses neighboring
        <bibl> and <quote> tags in a <cit> tag
        """
        # Check for and preserve XML declaration
        xml_declaration = ""
        if xml_string.lstrip().startswith("<?xml"):
            # Extract the XML declaration
            if "?>" in xml_string:
                xml_declaration = xml_string[: xml_string.index("?>") + 2]
                xml_string = xml_string[xml_string.index("?>") + 2 :]

        # Parse with lxml - wrap in root element to handle fragments
        try:
            root = etree.fromstring(f"<root>{xml_string}</root>")
        except etree.XMLSyntaxError as e:
            path_info = f"file {xml_path}" if xml_path else "XML string"
            logger.exception(f"lxml failed to process {path_info} with Exception: {e}")
            # NOTE: could change this to skip files that can't be parsed
            raise e

        # First pass: handle orphans (merge adjacent same-type tags)
        for element in root.iter():
            if cls._get_tag_name(element).lower() in cls.cit_elements:
                cls._handle_orphans(root, element)

        # Second pass: wrap bibl-quote pairs in cit tags
        # Need to re-iterate because tree was modified
        for element in root.iter():
            if cls._get_tag_name(element).lower() in cls.cit_elements:
                cls._handle_cit_elt(element)

        # Serialize back to string without XML declaration
        # Get the content inside our wrapper root element
        result_parts = [xml_declaration] if xml_declaration else []
        if root.text:
            result_parts.append(root.text)
        for child in root:
            # tostring includes the element's tail by default, so don't add it separately
            # xml_declaration=False ensures no <?xml...?> is added
            result_parts.append(
                etree.tostring(
                    child, encoding="unicode", method="xml", xml_declaration=False
                )
            )

        return "".join(result_parts)

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

        window_length = end - start
        # Use window-relative indexing
        center_ids = input_ids[:window_length]
        center_text = self.loader.tokenizer.decode(center_ids, skip_special_tokens=True)

        if isinstance(offset_mapping, torch.Tensor):
            offset_map = offset_mapping[:window_length].unsqueeze(0)
        else:
            offset_map = [offset_mapping[:window_length]]

        center_encoding = transformers.BatchEncoding(
            {
                "input_ids": center_ids.unsqueeze(0),
                "attention_mask": attention_mask[:window_length].unsqueeze(0),
                "offset_mapping": offset_map,
            }
        )
        return (
            center_text,
            center_encoding,
            [labels[i - start] for i in range(reliable_start, reliable_end)],
        )

    def strip_citation_tags(
        self, xml_content: str
    ) -> tuple[str, list[tuple[int, int, str, str]]]:
        """
        Strip citation tags and track their positions in the stripped text.

        Returns:
            (stripped_text, citations)
            where citations is a list of (start_char, end_char, tag_type, attributes)
            and positions are character offsets in stripped_text
        """

        # Check for and preserve XML declaration
        xml_declaration = ""
        if xml_content.lstrip().startswith("<?xml"):
            # Extract the XML declaration
            if "?>" in xml_content:
                xml_declaration = xml_content[: xml_content.index("?>") + 2]
                xml_content = xml_content[xml_content.index("?>") + 2 :]

        # Parse XML with lxml - wrap in root element to handle fragments
        try:
            root = etree.fromstring(f"<root>{xml_content}</root>")
        except etree.XMLSyntaxError:
            # If it fails, try with HTML parser which is more lenient
            from lxml import html

            root = html.fromstring(f"<root>{xml_content}</root>")

        # Build stripped text while tracking citation positions
        citations = []
        stripped_parts = []
        # Start position tracking after XML declaration (if present)
        current_pos = len(xml_declaration)

        def traverse(element: _Element) -> None:
            """Recursively traverse the parse tree, building stripped text."""
            nonlocal current_pos

            # Check if this element is a citation tag
            tag_name = CitationTagger._get_tag_name(element)
            is_citation = tag_name.lower() in (
                "bibl",
                "quote",
                "cit",
            )

            start_pos = None
            if is_citation:
                # Record the start position for citation
                start_pos = current_pos
            else:
                opening_tag = get_opening_tag(element)
                stripped_parts.append(opening_tag)
                current_pos += len(opening_tag)

            # Add element's text (text before first child)
            if element.text:
                stripped_parts.append(element.text)
                current_pos += len(element.text)

            # Process child elements
            for child in element:
                traverse(child)
                # Add tail text (text after child element)
                if child.tail:
                    stripped_parts.append(child.tail)
                    current_pos += len(child.tail)

            if is_citation:
                # Record the citation span
                end_pos = current_pos
                cit_attrs = get_attrs_as_string(element)
                citations.append((start_pos, end_pos, tag_name.upper(), cit_attrs))
            else:
                # For non-citation tags, include the closing tag
                closing_tag = f"</{tag_name}>"
                stripped_parts.append(closing_tag)
                current_pos += len(closing_tag)

        # Process root's children (unwrap our added root element)
        if root.text:
            stripped_parts.append(root.text)
            current_pos += len(root.text)

        for child in root:
            traverse(child)
            if child.tail:
                stripped_parts.append(child.tail)
                current_pos += len(child.tail)

        # Rebuild the stripped text, preserving the XML declaration if it existed
        result_parts = [xml_declaration] if xml_declaration else []
        result_parts.extend(stripped_parts)
        stripped_text = "".join(result_parts)

        # outer tags come before inner tags
        citations.sort(key=lambda x: (x[0], -x[1]))

        return stripped_text, citations
