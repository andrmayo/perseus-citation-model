import logging
import re
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, cast, Iterable

from lxml import etree
from lxml.etree import _Element, XMLSyntaxError
import torch
import transformers


import perscit_model.extraction.evaluate as evaluate
from perscit_model.extraction.data_loader import (
    ID2LABEL,
    ExtractionDataLoader,
    SPECIAL_TOKENS,
)
from perscit_model.extraction.inference import InferenceModel
from perscit_model.shared.xml_utils import get_encoding

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_MAPPING = {}
for spec_tok in SPECIAL_TOKENS:
    name, type = spec_tok.split("_")
    name, type = name[1:], type[:-1]
    SPECIAL_TOKENS_MAPPING[spec_tok] = (
        f"<{name.lower()}>" if type == "START" else f"</{name.lower()}>"
    )


class CitationTagger:
    cit_elements: Iterable[str] = ("quote", "bibl")

    def __init__(
        self,
        model_path: str | Path,
        window_size: int | None = None,
        stride: int | None = None,
        center: int | None = None,
        device: str | None = None,
        **kwargs,
    ):
        """
        Args:
            model_path: path to model to load
            window_size: defaults to max_length from config (to match training)
            stride: number of tokens to shift context window by, by default half of the window size
            center: defaults to stride, which ensures that every token winds up in center for some window
        """
        self.inference_model = InferenceModel(model_path, **kwargs)
        self.inference_model.model.eval()
        self.loader: ExtractionDataLoader = self.inference_model.loader
        # Use config max_length if window_size not specified
        self.window_size: int = (
            window_size if window_size is not None else self.loader.max_length
        )
        # this ensures that every token winds up in the reliable center of context window
        self.stride: int = stride if stride else self.window_size // 2
        self.center: int = center if center else self.stride
        # model in self.inference_model should have moved itself to GPU if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            self.inference_model.device = device
        self._batch_size = None

    # The approach here is to preprocess and tokenize the whole XML file
    # and then slide a window through the tokens
    def process_xml(
        self,
        xml_path: str | Path | Iterable[str] | Iterable[Path],
        overwrite: bool = True,
        remove_existing_citations: bool = False,
    ):
        """
        Process XML files to insert citation tags based on model predictions.

        Args:
            xml_path: path or Iterable of paths
            overwrite: if True, modify files in place; if False, create copies in processed/ subdirectory
        """
        if isinstance(xml_path, Path) and xml_path.is_dir():
            xml_path = list(xml_path.glob("*.xml"))
        elif not isinstance(xml_path, Iterable):
            xml_path = [xml_path]

        # Parallelize file processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self.process_xml_file,
                    path if isinstance(path, Path) else Path(path),
                    overwrite,
                    remove_existing_citations,
                )
                for path in xml_path
            ]

            # Wait for all to complete and raise any exceptions
            for future in futures:
                future.result()

    def process_xml_file(
        self,
        xml_path: Path,
        overwrite: bool = True,
        remove_existing_citations: bool = False,
    ) -> None:
        """
        Process a single XML file. Loads whole file but streams predictions.

        Args:
            xml_path: Path object giving path to XML file.
            overwrite: if True, modify file in place; if False, create copy in processed/ subdirectory
        """
        file_encoding = get_encoding(xml_path)
        if not file_encoding:
            file_encoding = "utf-8"
            logger.warning(
                f"No encoding declaration found for {xml_path.name}, defaulting to utf-8"
            )
        xml_content = xml_path.read_text(encoding=file_encoding)
        if remove_existing_citations:
            xml_content = evaluate.strip_xml_tags(xml_content)

        # Extract preamble (XML declaration, processing instructions) to preserve
        preamble, xml_content = CitationTagger._extract_preamble(xml_content)

        # IMPORTANT: Use truncation=False and padding=False to tokenize the entire XML file
        # The sliding window approach will handle chunking

        # Suppress tokenizer warning about byte fallback in fast tokenizers
        warnings.filterwarnings(
            "ignore",
            message=".*byte fallback.*",
            category=UserWarning,
            module="transformers.convert_slow_tokenizer",
        )
        encoding: transformers.BatchEncoding = self.loader.tokenizer(
            xml_content,
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        offset_mapping = encoding.offset_mapping[0]

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=xml_path.parent,
                delete=False,
                suffix=".tmp",
            ) as f:
                temp_path = Path(f.name)
                labels = self._get_labels(input_ids, attention_mask)

                # NOTE: offset_mapping is the only way to construct a minimally modified version
                # of the original text

                assert len(labels) == len(offset_mapping), (
                    "Labels and offsets aren't aligned"
                )
                new_xml: list[str] | str = []
                in_entity = False
                last_label = "O"
                last_end = 0
                warned_orphan_i = False  # Track if we've warned about orphan I- tokens

                for label, (start, end) in zip(labels, offset_mapping):
                    # This handles special tokens
                    if start == end:
                        continue

                    # this shouldn't do anything, but could handle overlaps if weird
                    # things happen with unicode normalization
                    start = max(start, last_end)

                    # Deal with invalid prediction (I- without preceding B-)
                    if not in_entity and label[0] == "I":
                        # Only warn once for consecutive orphan I- tokens
                        if not warned_orphan_i:
                            # Show context window around the invalid token
                            context_start = max(0, start - 200)
                            context_end = min(len(xml_content), end + 200)
                            context = xml_content[context_start:context_end]
                            # Mark the token position in context
                            token_in_context_start = start - context_start
                            token_in_context_end = end - context_start
                            marked_context = (
                                context[:token_in_context_start]
                                + ">>>"
                                + context[token_in_context_start:token_in_context_end]
                                + "<<<"
                                + context[token_in_context_end:]
                            )
                            # Collect labels for all tokens in context window
                            context_labels = []
                            for lbl, (s, e) in zip(labels, offset_mapping):
                                if s == e:  # Skip special tokens
                                    continue
                                if s >= context_start and e <= context_end:
                                    token_text = xml_content[s:e]
                                    context_labels.append(f"  {lbl:10} '{token_text}'")
                            labels_str = "\n".join(context_labels)
                            logger.warning(
                                f"Invalid: {label} without 'B-' label at position {start}-{end}\n"
                                f"Token: '{xml_content[start:end]}'\n"
                                f"Context (token marked with >>>...<<<):\n{marked_context}\n"
                                f"Labels in context:\n{labels_str}\n"
                            )
                            warned_orphan_i = True
                        # Promote orphan I- to B- to preserve the entity
                        label = "B-" + label[2:]

                    if label == "O":
                        if in_entity:
                            if last_label != "O":
                                new_xml.append(f"[{last_label}_END]")
                            in_entity = False

                        if start > last_end:
                            new_xml.append(xml_content[last_end:start])
                        new_xml.append(xml_content[start:end])
                        last_label = "O"
                        last_end = end
                        # Don't reset warned_orphan_i here - orphan I- tokens become O
                        # and we don't want to re-warn. Only reset on valid B- labels.
                        continue

                    label_type, label_name = label.split("-")
                    if label_type == "B":
                        if in_entity:
                            if last_label != "O":
                                new_xml.append(f"[{last_label}_END]")
                        if start > last_end:
                            new_xml.append(xml_content[last_end:start])
                        new_xml.extend(
                            [f"[{label_name}_START]", xml_content[start:end]]
                        )
                        in_entity = True
                        warned_orphan_i = False  # Reset for next potential orphan I- sequence
                    # label_type == "I"
                    else:
                        if label_name != last_label:
                            # Only warn once for consecutive invalid I- tokens
                            if not warned_orphan_i:
                                # Show context window around the invalid token
                                context_start = max(0, start - 200)
                                context_end = min(len(xml_content), end + 200)
                                context = xml_content[context_start:context_end]
                                token_in_context_start = start - context_start
                                token_in_context_end = end - context_start
                                marked_context = (
                                    context[:token_in_context_start]
                                    + ">>>"
                                    + context[token_in_context_start:token_in_context_end]
                                    + "<<<"
                                    + context[token_in_context_end:]
                                )
                                # Collect labels for all tokens in context window
                                context_labels = []
                                for lbl, (s, e) in zip(labels, offset_mapping):
                                    if s == e:  # Skip special tokens
                                        continue
                                    if s >= context_start and e <= context_end:
                                        token_text = xml_content[s:e]
                                        context_labels.append(f"  {lbl:10} '{token_text}'")
                                labels_str = "\n".join(context_labels)
                                logger.warning(
                                    f"Invalid: I-{last_label} followed by I-{label_name} at position {start}-{end}\n"
                                    f"Token: '{xml_content[start:end]}'\n"
                                    f"Context (token marked with >>>...<<<):\n{marked_context}\n"
                                    f"Labels in context:\n{labels_str}\n"
                                )
                                warned_orphan_i = True
                            # Close the previous entity if any
                            if last_label != "O":
                                new_xml.append(f"[{last_label}_END]")
                            # Promote orphan I- to B- to start a new entity
                            if start > last_end:
                                new_xml.append(xml_content[last_end:start])
                            new_xml.extend(
                                [f"[{label_name}_START]", xml_content[start:end]]
                            )
                            in_entity = True
                            last_label = label_name
                            last_end = end
                            continue
                        if start > last_end:
                            new_xml.append(xml_content[last_end:start])
                        new_xml.append(xml_content[start:end])

                    last_label = label_name
                    last_end = end

                # This really shouldn't be applicable, but just in case a snippet is passed
                # in that ends with a predicted tag:
                if in_entity and last_label != "O":
                    new_xml.append(f"[{last_label}_END]")

                new_xml = CitationTagger._fix_malformed_predictions(new_xml)

                new_xml = "".join(new_xml)
                tree = etree.fromstring(new_xml.encode(file_encoding))
                # Wrap adjacent bibl and quote pairs in cit
                for tag_name in ("bibl", "quote"):
                    # cast is safe because xpath returns a list for node seelection queries
                    for elem in cast(
                        list[_Element], tree.xpath(f".//*[local-name()='{tag_name}']")
                    ):
                        self._handle_cit_elt(elem)

                # encoding="unicode" gives str, ="utf-8" gives bytes
                new_xml = etree.tostring(tree, encoding="unicode", method="xml")
                new_xml = preamble + new_xml

                temp_path.write_text(new_xml, encoding=file_encoding)
            # atomic replace of original xml file
            if overwrite:
                temp_path.replace(xml_path)
            else:
                # create dir for processed xml files if overwrite = False
                # done here rather than in orchestration function since xml files
                # may live in different directories
                copy_dir = xml_path.parent / "processed"
                copy_dir.mkdir(exist_ok=True)
                temp_path.replace(copy_dir / xml_path.name)

        except XMLSyntaxError as e:
            logger.warning(f"Failed to process {xml_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to process {xml_path} {e}", exc_info=True)
        finally:
            # Clean up temp file
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

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
                self._batch_size = 64
            elif gpu_memory_gb >= 16:
                self._batch_size = 32
            elif gpu_memory_gb >= 8:
                self._batch_size = 16
            else:
                self._batch_size = 8
        else:
            self._batch_size = 4

        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int) -> None:
        print(f"Setting batch size to {new_batch_size}")
        self._batch_size = new_batch_size

    def _get_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int | None = None,
    ) -> list[str]:
        """
        Predictions from sliding windows over the tokenized XML.

        Args:
            input_ids: BatchEncoding attribute with token ids
            attention_mask: batch_encoding attribute
            offset_mapping: Tensor where offset_mapping[i] gives a Tensor of shape (seq_length, 2)
            xml_content: The original XML content text (for extracting windows)
            batch_size: dynamically set by default
        Returns:
            labels: list of BIO label strings
        """
        # Step 1: Create all windows
        # Step 2: Predict on windows in batches
        # Step 3: Merge predictions using reliable centers
        # Step 4: Convert to label strings

        def flatten_tensor(t: torch.Tensor, name: str) -> torch.Tensor:
            if t.ndim > 2:
                raise ValueError(
                    f"CitationTagger._get_labels received {name} Tensor with more than 2 dimensions"
                )
            # smooth out batched-format input tensors if applicable
            if t.ndim == 2:
                try:
                    t = t.squeeze(0)
                except RuntimeError:
                    raise ValueError(
                        f"It looks like CitationTagger._get_labels may have received batched input for {name}"
                    )
            return t

        input_ids = flatten_tensor(input_ids, "input_ids")
        attention_mask = flatten_tensor(attention_mask, "attention_mask")

        # Store original length before padding (for final output)
        original_n_tokens = len(input_ids)

        # Pad input sequence at start and end so all windows can use center-only
        # predictions (matching training where edge labels are masked)
        edge_margin = self.loader.max_length // 4
        pad_token_id = self.loader.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id configured")
        pad_token_id = cast(int, pad_token_id)

        front_pad_ids = torch.full((edge_margin,), pad_token_id, dtype=input_ids.dtype)
        front_pad_mask = torch.zeros(edge_margin, dtype=attention_mask.dtype)
        back_pad_ids = torch.full((edge_margin,), pad_token_id, dtype=input_ids.dtype)
        back_pad_mask = torch.zeros(edge_margin, dtype=attention_mask.dtype)

        input_ids = torch.cat([front_pad_ids, input_ids, back_pad_ids])
        attention_mask = torch.cat([front_pad_mask, attention_mask, back_pad_mask])

        # Step 1: Create windows

        if batch_size is None:
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size

        # Collect windows into batches
        windows = []
        last_reliable_end = 0  # Track where the last window's reliable center ended

        # Create sliding windows over padded sequence
        n_tokens = len(input_ids)
        for window_start in range(0, n_tokens, self.stride):
            window_end = min(window_start + self.window_size, n_tokens)

            # Uniform center calculation for all windows (no special cases)
            reliable_start = last_reliable_end
            reliable_end = min(reliable_start + self.stride, n_tokens)

            # For first window, start from edge_margin (skip front padding)
            if window_start == 0:
                reliable_start = edge_margin
                reliable_end = reliable_start + self.stride

            windows.append(
                {
                    "start": window_start,
                    "end": window_end,
                    "reliable_start": reliable_start,
                    "reliable_end": reliable_end,
                    "input_ids": input_ids[window_start:window_end],
                    "attention_mask": attention_mask[window_start:window_end],
                }
            )
            last_reliable_end = reliable_end

        # Step 2: Predict on all windows in batches
        all_window_predictions = []

        # Pad to max_length to match training preprocessing
        max_len = self.loader.max_length

        for batch_start in range(0, len(windows), self.batch_size):
            batch_windows = windows[batch_start : batch_start + self.batch_size]

            batch_input_ids = torch.stack(
                [
                    torch.nn.functional.pad(
                        w["input_ids"],
                        (0, max_len - w["input_ids"].shape[0]),
                        value=pad_token_id,
                    )
                    for w in batch_windows
                ]
            )

            batch_attention_mask = torch.stack(
                [
                    torch.nn.functional.pad(
                        w["attention_mask"],
                        (0, max_len - w["attention_mask"].shape[0]),
                        value=0,
                    )
                    for w in batch_windows
                ]
            )

            # Move to device and predict
            batch_input_ids = batch_input_ids.to(self.device)
            batch_attention_mask = batch_attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.inference_model.model(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask
                )

            # Use CRF Viterbi decode if available, otherwise argmax
            if hasattr(self.inference_model.model, "decode"):
                # CRF decode returns list of lists (variable length per sequence)
                byte_mask = batch_attention_mask.byte()
                # CRF requires first timestep to be unmasked for decoding to start.
                # This is safe because predictions for padded positions are ignored
                # anyway (we only use reliable_start:reliable_end predictions).
                byte_mask[:, 0] = 1
                predictions = cast(Callable, self.inference_model.model.decode)(
                    outputs.logits, byte_mask
                )
                # Convert to tensor for consistent handling below
                max_pred_len = max(len(p) for p in predictions)
                predictions = torch.tensor(
                    [p + [0] * (max_pred_len - len(p)) for p in predictions],
                    device="cpu",
                )
            else:
                # predictions has dimensions [batch_size, max_len]
                predictions = outputs.logits.argmax(dim=-1)

            # Store predictions for each window (unpadded)
            for i, window in enumerate(batch_windows):
                window_length = window["end"] - window["start"]
                all_window_predictions.append(predictions[i, :window_length].cpu())

        # Step 3: Merge predictions using reliable centers
        # Build predictions for the full padded sequence
        padded_predictions = [None] * n_tokens

        for window, window_preds in zip(windows, all_window_predictions):
            # index within window
            reliable_start_idx = window["reliable_start"] - window["start"]
            reliable_end_idx = window["reliable_end"] - window["start"]

            # Clamp indices to actual prediction length (for short sequences
            # where reliable region may extend beyond window bounds)
            reliable_end_idx = min(reliable_end_idx, len(window_preds))
            actual_reliable_end = window["start"] + reliable_end_idx

            # Copy reliable predictions
            padded_predictions[window["reliable_start"] : actual_reliable_end] = [
                window_preds[i].item()
                for i in range(reliable_start_idx, reliable_end_idx)
            ]

        # Extract only the original tokens (skip front and back padding)
        final_predictions = padded_predictions[
            edge_margin : edge_margin + original_n_tokens
        ]

        def get_pred_label(pred_val: int | None) -> str:
            if pred_val is None:
                raise ValueError(
                    "Issue with window logic in CitationTagger._get_labels"
                )
            return ID2LABEL[pred_val]

        # Step 4: convert prediction indices to label strings
        return [get_pred_label(pred) for pred in final_predictions]

    @staticmethod
    def _handle_tag_stack(
        new_tag: str,
        tag_stack: list[tuple[str, int]],
        prior_xml_pieces: list[str],
        new_tag_is_prediction: bool = True,
    ) -> tuple[list[tuple[str, int]], list[str], bool]:
        """
        Repairs malformed XML nesting by reordering misplaced closing tags.

        Semantics:
        - Opening tags are appended normally, or to the right if they fall
          within the angle brackets of a tag.
        - Closing tags are either appended if not within tag itself, otherwise
          are inserted to left of tag.
        - If closing tag does is not well-formed with prior opening tag,
          the closing tag tag is moved left until it forms valid XML
        - NOTE: implementation is a bit awkward, in that _handle_tag_stack
          sometimes mutates prior_xml_pieces, and sometimes doesn't.
          If new_tag still needs to be appended, the third element
          in the returned tuple is True.

        tag_stack stores (opening_tag_text, index_in_prior_xml_pieces).
        """

        def tag_name(tag: str) -> str:
            m = re.match(r"</?\s*([A-Za-z_][\w:.-]*)", tag)
            if not m:
                raise ValueError(f"Invalid tag: {tag}; new_tag: {new_tag}")
            return m.group(1)

        # ---- Self-closing tag ----
        if new_tag.endswith("/>"):
            return tag_stack, prior_xml_pieces, True

        # ---- Closing tag ----
        if new_tag.startswith("</"):
            name = tag_name(new_tag)
            # We're not interested in repairing any tags
            # but the ones we're inserting
            if not new_tag_is_prediction:
                if len(tag_stack) == 0:
                    raise ValueError("XML is invalid")
                if tag_stack[-1][0][1] != "/" and tag_name(tag_stack[-1][0]) == name:
                    del tag_stack[-1]
                    return tag_stack, prior_xml_pieces, True
                else:
                    tag_stack.append((new_tag, len(prior_xml_pieces)))
                    # if we have <a><predicted_a></a></predicted_a>
                    # then the current tag_stack could be
                    # <a><predicted_a></a>, i.e. invalid, and needs to be
                    # repaired by moving <predicted_a> when we encounter </predicted_a>,
                    # i.e. <a></a><predicted_a></predicted_a>
                return tag_stack, prior_xml_pieces, True

            # Fast path: well-nested -> emit normally
            if tag_stack and tag_name(tag_stack[-1][0]) == name:
                tag_stack.pop()
                return tag_stack, prior_xml_pieces, True

            # Find matching opening tag in the stack
            for idx in range(len(tag_stack) - 1, -1, -1):
                if tag_name(tag_stack[idx][0]) == name:
                    break
            else:
                logger.warning(
                    f"No opening tag found for {new_tag} while repairing malformed predictions; dropping it"
                )
                return tag_stack, prior_xml_pieces, False

            # Move opening tag right until XML is well nested
            open_tag, open_pos = tag_stack[idx]
            # The blocking tag is the LAST tag opened after the opening tag
            # (i.e., the last element in the tag_stack)
            _, insertion_pos = tag_stack[-1]
            insertion_pos += 1

            prior_xml_pieces.insert(insertion_pos, open_tag)
            prior_xml_pieces.append(new_tag)

            # check if previously ophaned opening tag was separating matching tags
            while (
                idx > 0
                and idx < (len(tag_stack) - 1)
                and tag_stack[idx - 1][0][1] != "/"
                and tag_stack[idx + 1][0][1] == "/"
            ):
                prev_name, next_name = (
                    tag_name(tag_stack[idx - 1][0]),
                    tag_name(tag_stack[idx + 1][0]),
                )
                if prev_name == next_name:
                    del tag_stack[idx - 1]
                    idx -= 1
                    del tag_stack[idx + 1]

            # Remove the matched opening tag from the stack and its prior position from prior_xml_pieces
            del tag_stack[idx]
            del prior_xml_pieces[open_pos]

            # Adjust stored positions for tags that occur after the insertion point
            for i in range(len(tag_stack)):
                tag, pos = tag_stack[i]
                if open_pos < pos < insertion_pos:
                    tag_stack[i] = (tag, pos - 1)

            # closing tag already emitted
            return tag_stack, prior_xml_pieces, False

        # ---- Opening tag ----
        else:
            tag_stack.append((new_tag, len(prior_xml_pieces)))
            return tag_stack, prior_xml_pieces, True

    @staticmethod
    def _fix_malformed_predictions(xml: list[str] | str) -> list[str]:
        if isinstance(xml, list):
            xml = "".join(xml)

        pattern = re.compile(
            r"("
            r"<!--[\s\S]*?-->"  # XML comments
            r"|<!"
            r'(?:[^>"\']+|"[^"]*"|\'[^\']*\')+'
            r">"
            r"|<\?"
            r'(?:[^>"\']+|"[^"]*"|\'[^\']*\')+'
            r"\?>"
            r"|<"
            r'(?:[^>"\']+|"[^"]*"|\'[^\']*\')+'
            r">"
            r")"
        )

        re_tokenized = re.split(pattern, xml)

        # Now, every tag, even if it has e.g. [BIBL_START] invalidly inside it, is
        # its own element, and each tag-like element (e.g. CDATA), and then each
        # content between two tags or tag-like elements

        tag_stack: list[tuple[str, int]] = []
        xml_with_preds = []
        special_token_pattern = r"(\[[A-Z]+_(?:START|END)\])"
        for token in re_tokenized:
            if "_START]" in token or "_END]" in token:
                # deal with tag-like structures
                if token[0] == "<" and token[-1] == ">":
                    # deal with non-tag structures with <>
                    opening_special_tags = []
                    for special_token in re.findall(special_token_pattern, token):
                        special_tag = SPECIAL_TOKENS_MAPPING[special_token]
                        token = token.replace(special_token, "")
                        # opening tags are easy
                        if special_tag[1] != "/":
                            opening_special_tags.append(special_tag)
                        # closing tags trigger nesting checks
                        else:
                            # handle case where both opening and closing special tag are in regular tag
                            corres_tag = f"<{special_tag[2:-1]}>"
                            if corres_tag in opening_special_tags:
                                for i in range(len(opening_special_tags) - 1, -1, -1):
                                    if opening_special_tags[i] == corres_tag:
                                        del opening_special_tags[i]
                                        break
                                # skip special tags entirely contained within regular tag
                                continue
                            # Now handle case where only closing tag is within regular tag
                            tag_stack, xml_with_preds, emit = (
                                CitationTagger._handle_tag_stack(
                                    special_tag, tag_stack, xml_with_preds
                                )
                            )
                            if emit:
                                xml_with_preds.append(special_tag)

                    # if <...> entity is a comment etc., don't pass it through tag handling logic
                    if token[1] == "!" or token[1] == "?":
                        xml_with_preds.append(token)
                    # First emit the regular tag
                    else:
                        tag_stack, xml_with_preds, emit = (
                            CitationTagger._handle_tag_stack(
                                token, tag_stack, xml_with_preds, False
                            )
                        )
                        if emit:
                            xml_with_preds.append(token)

                    # Then emit opening special tags
                    for open_tag in opening_special_tags:
                        tag_stack, xml_with_preds, emit = (
                            CitationTagger._handle_tag_stack(
                                open_tag, tag_stack, xml_with_preds
                            )
                        )
                        if emit:
                            xml_with_preds.append(open_tag)

                    # token has now been stripped of all special tokens
                    continue
                # Non-tag-like structures
                split_by_spec = re.split(special_token_pattern, token)
                for tkn in split_by_spec:
                    is_spec_tkn = False
                    if tkn in SPECIAL_TOKENS_MAPPING:
                        is_spec_tkn = True
                    tkn = SPECIAL_TOKENS_MAPPING.get(tkn, tkn)
                    if len(tkn) > 2 and tkn[0] == "<" and tkn[-1] == ">":
                        tag_stack, xml_with_preds, emit = (
                            CitationTagger._handle_tag_stack(
                                tkn, tag_stack, xml_with_preds, is_spec_tkn
                            )
                        )
                        if emit:
                            xml_with_preds.append(tkn)
                    else:
                        xml_with_preds.append(tkn)

            else:
                # Handle tokens without special tokens (much simpler case)
                if len(token) > 2 and (token[0] == "<" and token[-1] == ">"):
                    # check that not merely tag-like entity
                    if not (token[1] == "!" or token[1] == "?"):
                        tag_stack, xml_with_preds, emit = (
                            CitationTagger._handle_tag_stack(
                                token, tag_stack, xml_with_preds, False
                            )
                        )
                        if emit:
                            xml_with_preds.append(token)
                    else:
                        xml_with_preds.append(token)
                else:
                    xml_with_preds.append(token)

        return xml_with_preds

    @staticmethod
    def _get_tag_name(element: _Element) -> str:
        """Safely get tag name from lxml element, handling QNames and other types."""
        try:
            return etree.QName(element).localname
        except (ValueError, TypeError):
            # Fallback for non-standard tag types
            return str(element.tag) if hasattr(element, "tag") else ""

    @staticmethod
    def _extract_preamble(xml_string: str) -> tuple[str, str]:
        """Extract XML declaration and processing instructions from the start of XML.

        Returns:
            (preamble, remaining_xml) where preamble contains all leading processing instructions
        """
        # Match all processing instructions at the start (including <?xml...?>)
        match = re.match(r"(\s*(?:<\?.*?\?>\s*)+)", xml_string)
        if match:
            preamble = match.group(1)
            remaining = xml_string[len(preamble) :]
            return preamble, remaining
        return "", xml_string

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
