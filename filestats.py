import os
import re
import argparse
from typing import Pattern, List, Tuple, Optional
import pathspec

def load_gitignore_spec(directory: str, gitignore_path: str) -> Optional[pathspec.PathSpec]:
	"""Loads gitignore patterns from a file."""
	if not os.path.isabs(gitignore_path):
		gitignore_path = os.path.join(directory, gitignore_path)

	if not os.path.isfile(gitignore_path):
		print(f"Warning: .gitignore file not found at {gitignore_path}")
		return None

	with open(gitignore_path, 'r') as f:
		return pathspec.PathSpec.from_lines('gitwildmatch', f)

def get_matching_filenames(directory: str, pattern: str, recursive: bool = False, spec: Optional[pathspec.PathSpec] = None) -> List[str]:
	"""
	Finds filenames in the given directory that match the provided regex pattern.

	Args:
		directory (str): Path to the directory.
		pattern (str): Regular expression pattern to match filenames.
		recursive (bool): If True, search recursively in subdirectories.
		spec (pathspec.PathSpec, optional): A pathspec object for ignoring files.

	Returns:
		List[str]: A list of matching filenames.
	"""
	regex: Pattern = re.compile(pattern)
	matching_files: List[str] = []
	if recursive:
		for dirpath, dirnames, fnames in os.walk(directory, topdown=True):
			if spec:
				# Get paths relative to the search directory for spec matching
				relative_dirpath = os.path.relpath(dirpath, directory)
				if relative_dirpath == '.':
					relative_dirpath = ''

				# Filter directories in-place
				original_dirnames = list(dirnames)
				dirnames[:] = [d for d in original_dirnames if not spec.match_file(os.path.join(relative_dirpath, d, ''))]

				# Filter files
				fnames = [f for f in fnames if not spec.match_file(os.path.join(relative_dirpath, f))]

			for fname in fnames:
				if regex.match(fname):
					full_path = os.path.join(dirpath, fname)
					relative_path = os.path.relpath(full_path, directory)
					matching_files.append(relative_path)
		return matching_files

	try:
		filenames = os.listdir(directory)
	except OSError as e:
		print(f"Error reading directory: {e}")
		return []

	if spec:
		filenames = [f for f in filenames if not spec.match_file(f)]

	matching_files = [fname for fname in filenames if regex.match(fname) and os.path.isfile(os.path.join(directory, fname))]
	return matching_files

def count_content_matches(directory: str, pattern: str, recursive: bool = False, spec: Optional[pathspec.PathSpec] = None) -> Tuple[int, List[str], List[str]]:
	"""
	Counts regex matches in file contents and lists files with matches.

	Args:
		directory (str): Path to the directory to search.
		pattern (str): Regular expression pattern to match in content.
		recursive (bool): If True, search recursively in subdirectories.
		spec (pathspec.PathSpec, optional): A pathspec object for ignoring files.

	Returns:
		Tuple[int, List[str], List[str]]: A tuple containing the total number of matches,
		a list of filenames with at least one match, and a list of all matches.
	"""
	regex: Pattern = re.compile(pattern)
	total_matches = 0
	files_with_matches: List[str] = []
	all_matches: List[str] = []

	files_to_search = []
	if recursive:
		for dirpath, dirnames, fnames in os.walk(directory, topdown=True):
			if spec:
				relative_dirpath = os.path.relpath(dirpath, directory)
				if relative_dirpath == '.':
					relative_dirpath = ''

				original_dirnames = list(dirnames)
				dirnames[:] = [d for d in original_dirnames if not spec.match_file(os.path.join(relative_dirpath, d, ''))]

				fnames = [f for f in fnames if not spec.match_file(os.path.join(relative_dirpath, f))]

			for fname in fnames:
				files_to_search.append(os.path.join(dirpath, fname))
	else:
		try:
			filenames = os.listdir(directory)
			if spec:
				filenames = [f for f in filenames if not spec.match_file(f)]

			for fname in filenames:
				full_path = os.path.join(directory, fname)
				if os.path.isfile(full_path):
					files_to_search.append(full_path)
		except OSError as e:
			print(f"Error reading directory: {e}")
			return 0, [], []

	for filepath in files_to_search:
		try:
			with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
				content = f.read()

				# Use finditer to get match objects with positions
				file_has_match = False
				for match in regex.finditer(content):
					if not file_has_match:
						relative_path = os.path.relpath(filepath, directory)
						files_with_matches.append(relative_path)
						file_has_match = True

					total_matches += 1

					# Extend the match to include following alphanumeric and underscore characters
					original_match_str = match.group(0)
					end_pos = match.end()

					extended_match = original_match_str
					while end_pos < len(content) and (content[end_pos].isalnum() or content[end_pos] == '_'):
						extended_match += content[end_pos]
						end_pos += 1

					all_matches.append(extended_match)

		except Exception:
			# Skip files that can't be read (e.g., binary files, permission errors)
			pass

	return total_matches, files_with_matches, all_matches

def main():
	parser = argparse.ArgumentParser(description="Count or list filenames or content matches for a regex in a directory.")
	parser.add_argument("directory", help="Path to the directory")
	parser.add_argument("pattern", help="Regular expression pattern to match filenames or content")
	parser.add_argument("-l", "--list", action="store_true", help="List matching filenames instead of counting them")
	parser.add_argument("-r", "--recursive", action="store_true", help="Recursively search in child directories")
	parser.add_argument("-c", "--content", action="store_true", help="Search file contents instead of filenames")
	parser.add_argument("-u", "--unique", action="store_true", help="Count or list unique content matches. Requires -c.")
	parser.add_argument("-g", "--use-gitignore", action="store_true", help="Use .gitignore file to filter results. When --gitignore-path is not passed, defaults to .gitignore in the search directory.")
	parser.add_argument("--gitignore-path", default=".gitignore", help="Supply a path to .gitignore, relative to the search directory. Requires -g.")
	args = parser.parse_args()

	if args.unique and not args.content:
		parser.error("-u/--unique requires -c/--content.")

	spec = None
	if args.use_gitignore:
		spec = load_gitignore_spec(args.directory, args.gitignore_path)

	if args.content:
		total_matches, files_with_matches, all_matches = count_content_matches(args.directory, args.pattern, args.recursive, spec)
		if args.unique:
			unique_matches = sorted(list(set(all_matches)))
			if args.list:
				for match in unique_matches:
					print(match)
			else:
				print(f"Number of unique content matches: {len(unique_matches)}")
		else:
			if args.list:
				for fname in files_with_matches:
					print(fname)
			else:
				print(f"Number of content matches: {total_matches}")
	else:
		matching_files = get_matching_filenames(args.directory, args.pattern, args.recursive, spec)

		if args.list:
			for fname in matching_files:
				print(fname)
		else:
			print(f"Number of matching files: {len(matching_files)}")

if __name__ == "__main__":
	main()
