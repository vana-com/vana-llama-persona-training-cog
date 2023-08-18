import requests
from hashlib import sha512
import re
import os
import re
from urllib.parse import urlparse, unquote


def get_path_and_filename(url):
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")
    filename = unquote(path.split("/")[-1])
    dir_path = "/".join(path.split("/")[:-1])
    return dir_path, filename


class FileManager:
    def __init__(self, download_dir=".") -> None:
        self.download_dir = download_dir
        self.file_urls = {}

    def download_file(self, url, preserve_structure=True, download_dir=None):
        if download_dir is None:
            download_dir = self.download_dir

        # Did I pass in a path instead of URL?
        if os.path.isfile(url):
            return url

        # Did I already download this file?
        if self.file_urls.get(url, None) is not None:
            return self.file_urls[url]

        # Start by Presuming that the URL is a path to a file
        path, filename = get_path_and_filename(url)
        if (
            bool(path)
            and bool(filename)
            and os.path.isfile(os.path.join(path, filename))
        ):
            return os.path.join(path, filename)

        # If flat_structure is True, ignore the nested path and download to the root directory
        if not preserve_structure:
            path = ""

        path = os.path.join(download_dir, path)
        os.makedirs(path, exist_ok=True)

        response = requests.get(url)
        response.raise_for_status()

        if "Content-Disposition" in response.headers.keys():
            filename = re.search(
                r'filename[^"]*"([^"]+)"', response.headers["Content-Disposition"]
            ).group(1)

        saved_filename = os.path.join(path, filename)

        with open(saved_filename, "wb") as f:
            f.write(response.content)

        self.file_urls[url] = os.path.join(path, filename)

        return os.path.join(path, filename)
