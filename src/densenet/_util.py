import zipfile
import requests

def download_file(url, filename):
  """Downloads a file from the specified URL and saves it with the given filename."""
  response = requests.get(url, stream=True)
  response.raise_for_status()  # Raise an exception for failed requests

  with open(filename, 'wb') as file:
    for chunk in response.iter_content(1024):
      # Download data in chunks to handle large files efficiently
      if chunk:  # filter out keep-alive new chunks
        file.write(chunk)

  print(f"Downloaded file: {filename}")


def unzip_file(zip_filepath, target_dir):
  """
  Extracts all files from a zip file to a specified directory.

  Args:
      zip_filepath (str): Path to the zip file.
      target_dir (str): Path to the target directory for extraction.
  """
  with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
    zip_ref.extractall(target_dir)

if __name__ == "__main__":
    url = "https://www.example.com/file.txt"
    filename = "downloaded_file.txt"

    download_file(url, filename)
