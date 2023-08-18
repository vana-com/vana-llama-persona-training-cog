#!/usr/bin/env python3

from typing import List
import os
import time
import unicodedata
import json
import re
import random
import argparse

SYSTEM_PROMPT = "Complete the chat below."


class ChatDataProcessor:
    def __init__(self, train_split=0.9, validate_split=None, filter_speaker=None, mask=False, no_overlap=False, final_format='replicate'):
        self.filter_speaker = filter_speaker
        self.mask = mask
        self.no_overlap = no_overlap
        self.final_format = final_format
        self.train_split = train_split
        if validate_split is None:
            self.validate_split = 1.0 - self.train_split
        else:
            self.validate_split = validate_split
        self.test_split = 1.0 - (self.train_split + self.validate_split)

    def remove_non_printable(self, text):
        text = text.replace('\r', '')
        text = text.replace('\u200e', '')
        return ''.join(char for char in text if char.isprintable())

    def format_text(self, text, old_timestamp_format):
        if self.mask:
            speaker = self.extract_speaker(text, old_timestamp_format)
            return f"{speaker}: {'*' * 5}"
        else:
            return self.remove_non_printable(text.split(" - ", 1)[1].strip()) if old_timestamp_format else self.remove_non_printable(text.split("] ", 1)[1].strip())

    def extract_speaker(self, line, old_timestamp_format):
        return line.split(" - ")[1].split(":")[0] if old_timestamp_format else line.split("] ")[1].split(":")[0]

    def generate_jsonl_line(self, occurrence, old_timestamp_format):
        msg = ["\n".join([self.format_text(text, old_timestamp_format) for text in occurrence[i]]) for i in range(4)]
        prompt = f'<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{msg[0]} [/INST] {msg[1]} </s><s>[INST] {msg[2]} [/INST]'
        completion = msg[3]

        if self.final_format == 'colab':
            example = {"text": f"{prompt} {completion}"}
        else:  # 'replicate' format
            example = {"prompt": prompt, "completion": f'{completion} '}

        return json.dumps(example, ensure_ascii=False)

    def process_directory(self, directory_path: str):
        input_files = [os.path.join(directory_path, file) for file in os.listdir(
            directory_path) ] # For now, assume all files are valid # if file.endswith('.txt')]
        return self.process_files(input_files)

    def process_files(self, input_files: List[str]):
        all_lines = []
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as file:
                raw_lines = file.readlines()
                raw_lines = [self.remove_non_printable(unicodedata.normalize("NFKC", line)) for line in raw_lines]

            lines = self.group_lines(raw_lines)
            message_bundles = self.extract_message_bundles(lines)

            jsonl_lines = self.generate_jsonl_from_bundles(message_bundles)
            all_lines.extend(jsonl_lines)

        return self.write_output_files(all_lines)

    def group_lines(self, raw_lines):
        lines = []
        current_message = ""
        for line in raw_lines:
            if re.match(r"\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} (AM|PM) - |\[\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2} (AM|PM)\] ", line):
                if current_message:
                    lines.append(current_message)
                    current_message = ""
                current_message = line.strip()
            else:
                current_message += " " + line.strip()
        if current_message:
            lines.append(current_message)
        return lines

    def extract_message_bundles(self, lines):
        message_bundles = []
        current_bundle = []
        last_speaker = None
        for line in lines:
            speaker = self.extract_speaker(line, " - " in line)
            if speaker and speaker != last_speaker:
                if current_bundle:
                    message_bundles.append(current_bundle)
                    current_bundle = []
            if speaker:
                current_bundle.append(line)
            last_speaker = speaker
        if current_bundle:
            message_bundles.append(current_bundle)
        return message_bundles

    def generate_jsonl_from_bundles(self, message_bundles):
        jsonl_lines = []
        i = 0
        while i <= len(message_bundles) - 4:
            old_timestamp_format = " - " in message_bundles[0][0]
            speakers = [self.extract_speaker(message_bundles[i + j][0], old_timestamp_format) for j in range(4)]

            # Filtering logic to skip over bundles that don't match the filter speaker
            if self.filter_speaker and self.filter_speaker != speakers[3]:
                i += 1
                continue

            if speakers[0] == speakers[2] and speakers[1] == speakers[3]:
                occurrence = [message_bundles[i + j] for j in range(4)]
                jsonl_lines.append(self.generate_jsonl_line(occurrence, old_timestamp_format))
                if self.no_overlap:
                    i += 4
                else:
                    i += 1
            else:
                i += 1
        return jsonl_lines

    def write_output_files(self, all_lines):
        random.shuffle(all_lines)
        current_time_millis = int(time.time() * 1000)

        train_index = int(self.train_split * len(all_lines))
        validate_index = train_index + int(self.validate_split * len(all_lines))

        # TODO: Make sure this correctly chops up all_lines
        train_lines = all_lines if self.train_split == 1.0 else all_lines[:train_index]
        validate_lines = None if self.train_split == 1.0 else all_lines[train_index:validate_index + (1 if self.test_split == 0 and validate_index < len(all_lines) else 0)]
        test_lines = all_lines[validate_index:] if self.test_split > 0 else None

        ext = '.jsonl' if self.final_format == 'colab' else '.json'

        train_file_name = f"train_{current_time_millis}{ext}"
        validate_file_name = f"validate_{current_time_millis}{ext}" if self.validate_split > 0 else None
        test_file_name = f"test_{current_time_millis}{ext}" if self.test_split > 0 else None

        if self.final_format == 'colab':
            self.write_jsonl_file(train_file_name, train_lines)
            if validate_lines is not None:
                self.write_jsonl_file(validate_file_name, validate_lines)
            if test_lines:
                self.write_jsonl_file(test_file_name, test_lines)
        else:  # 'replicate' format
            self.write_json_file(train_file_name, train_lines)
            if validate_lines is not None:
                self.write_json_file(validate_file_name, validate_lines)
            if test_lines:
                self.write_json_file(test_file_name, test_lines)

        return train_file_name, validate_file_name, test_file_name

    def write_jsonl_file(self, output_file, lines):
        with open(output_file, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line + '\n')

    def write_json_file(self, output_file, lines):
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('[\n')
            file.write(',\n'.join(lines))
            file.write('\n]')


def main():
    parser = argparse.ArgumentParser(description='Process chat data into train and validate JSON/JSONL files.')
    parser.add_argument('input_files', type=str, nargs='+', help='Paths to the input text files.')
    parser.add_argument('--filter-speaker', type=str, help='Filter by speaker name.')
    parser.add_argument('--mask', action='store_true', help='Mask the text.')
    parser.add_argument('--no-overlap', action='store_true', help='Generate non-overlapping examples.')
    parser.add_argument('--final-format', type=str, choices=['replicate', 'colab'], default='replicate', help='Final format.')
    parser.add_argument('--train-split', type=float, default=0.9, help='Percentage of data for training.')
    parser.add_argument('--validate-split', type=float, help='Percentage of data for validation.')

    args = parser.parse_args()

    processor = ChatDataProcessor(
        train_split=args.train_split,
        validate_split=args.validate_split,
        filter_speaker=args.filter_speaker,
        mask=args.mask,
        no_overlap=args.no_overlap,
        final_format=args.final_format
    )
    train_file_name, validate_file_name, test_file_name = processor.process_files(args.input_files)

    print(train_file_name)
    if validate_file_name:
        print(validate_file_name)
    if test_file_name:
        print(test_file_name)

if __name__ == "__main__":
    main()

