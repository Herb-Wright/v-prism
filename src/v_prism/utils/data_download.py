import requests
from requests import Response
from tqdm import tqdm
import tarfile
from typing import Optional

# --------------------------------------------------------
# these are from (altered slightly): 
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(
	id: str, 
	destination: str, 
	verbose: bool = False,
) -> None:	
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	
	# make sure our response went through
	for key, value in response.cookies.items():
		print(f'{key}: {value}')

	params = { 'id' : id, 'confirm' : 1 }
	response = session.get(URL, params = params, stream = True)

	if response.status_code == 429:
		raise Exception('You have sent too many requests, try again later')

	_save_response_content(response, destination, verbose=verbose)    

def _save_response_content(response: Response, destination: str, verbose: bool = False) -> None:
	CHUNK_SIZE = 32768
	if verbose:
		total_size_in_bytes= int(response.headers.get('content-length', 0))
		bar = tqdm(
			response.iter_content(CHUNK_SIZE), 
			total=((total_size_in_bytes // CHUNK_SIZE)),
			desc='downloading file from google drive'
		)
	else:
		bar = response.iter_content(CHUNK_SIZE)
	with open(destination, "wb+") as f:
		for chunk in bar:
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

# --------------------------------------------------------

def unzip_tar_file(tar_file_path: str, out_dir: str, verbose = True, num: Optional[int] = None) -> None:
	with tarfile.open(tar_file_path, 'r') as f:
		if verbose:
			bar = tqdm(iterable=f, desc='extracting file', total=num)
		else:
			bar = f
		for member in bar:
			f.extract(member, path=out_dir)


