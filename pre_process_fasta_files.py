from typing import List, Tuple
import pathlib, os
import glob
import zipfile
import pickle

import numpy as np
import pandas as pd

import string
import itertools

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


from argparse import ArgumentParser
import sys
import zipfile


deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def read_sequence(filename: str) -> Tuple[str, str]:
	"""Reads the first (reference) sequences from a fasta or MSA file."""
	record = next(SeqIO.parse(filename, "fasta"))
	return record.description, str(record.seq)


def remove_insertions(sequence: str) -> str:
	"""Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
	return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
	"""Reads the first nseq sequences from an MSA file, automatically removes insertions."""
	if nseq == -1:
		nseq = len([elem.id for elem in SeqIO.parse(filename, "fasta")])
	return [
		(record.description, remove_insertions(str(record.seq)))
		for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
	]

# extracts files from a zip file

def extract_zip_file (zfile_name, extract_to_path):
	if zipfile.is_zipfile(zfile_name):
		zf_handle = zipfile.ZipFile(zfile_name)
		try:
			zf_handle.extractall(extract_to_path)
		except Exception as e:
			print (e)
			return False
	else:
		print (f"Not a valid zip file : {zfile_name}")
		return False

	return True

# provide summary of several fasta files in a given directory
# helps to see what is in the MSA files - how many species, msa/per species etc

def eda_on_msa_data (folder_name):
		os.chdir(folder_name)
		# sort file names so (hopefully) we can catch related MSA from the name
		data_dict_list = []
		for ind_file in sorted(glob.glob("*.fasta")):
				data_dict = {}
				proc_msa_data = read_msa(ind_file, -1)
				species_names, seq_idx, species_count = np.unique(
				[get_species_name(rec[0]) for rec in proc_msa_data[1: ]],
				return_inverse=True,
				return_counts=True,
			)
	
	# seq_idx is off by 1 - first record in the file is not a MSA sequence
		data_dict = { "name": proc_msa_data[0][0],
				 "total_msa_seqs" : len(proc_msa_data) -1,
					"num_species" : len(species_names),
					"species_names_list" : species_names,
					"seq_ids_list" : seq_idx,
					"species_msa_count_list" : species_count}
		data_dict_list.append(data_dict)
		print (proc_msa_data[0][0], len(proc_msa_data) -1, len(species_names), np.average(species_count)  )
		return proc_msa_data, data_dict_list

# extract MSA data from a single fasta file
def ext_msa_from_fasta_file (fasta_file_name):
	"""
	process a single fasta file
	"""
	data_dict = {}
	proc_msa_data = read_msa(fasta_file_name, -1)
	species_names, seq_idx, species_count = np.unique(
		[get_species_name(rec[0]) for rec in proc_msa_data[1: ]],
		return_inverse=True,
		return_counts=True,
	)
	
	# seq_idx is off by 1 - first record in the file is not a MSA sequence
	data_dict = { "name": proc_msa_data[0][0],
				 "total_msa_seqs" : len(proc_msa_data) -1,
					"num_species" : len(species_names),
					"species_names_list" : species_names,
					"seq_ids_list" : seq_idx,
					"species_msa_count_list" : species_count}

	print (proc_msa_data[0][0], len(proc_msa_data) -1, len(species_names), np.average(species_count)  )
	return proc_msa_data, data_dict

# process two fasta files, DiffPALM requires MSA/Species to match
# following will add padded data  to match MSA / species

def process_two_fasta_files (file1, file2, out_dir):
	msa_1, summary_1 = ext_msa_from_fasta_file (file1)
	msa_2, summary_2 = ext_msa_from_fasta_file (file2)

	# orig (chain) sequence saved. This affects how we parse MSAs
	# DiffPALM expects only MSA - not the original sequence
	#file1_processed_msa_list = [msa_1[0]]
	#file2_processed_msa_list = [msa_2[0]]

	# original sequence not in the list
	# should keep track of it separately, only MSAs in the list
	file1_processed_msa_list = []
	file2_processed_msa_list = []

	seq_1_size = len(msa_1[1][1])
	seq_2_size = len(msa_2[1][1])
	common_species_list = set(summary_1["species_names_list"]).intersection(set(summary_2["species_names_list"])) 
	# extract relevant data from summary dict for easy access during the for loop
	sp_names_array_1 = summary_1["species_names_list"]
	sp_names_array_2 = summary_2["species_names_list"]
	
	# converting to numpy array to find all MSA indices of an species in a list
	# during iteration
	sp_seq_idx_1 = summary_1["seq_ids_list"]
	sp_seq_idx_2 = summary_2["seq_ids_list"]
	
	for sp in common_species_list:
		sp_idx_1 = np.where(sp_names_array_1 == sp)[0]
		sp_idx_2 = np.where(sp_names_array_2 == sp)[0]
	
		# get the index of MSA with for this species in the file
		# where returns np.array, extract the list from array with [0]
		msa_idx_1 = np.where(sp_seq_idx_1 == sp_idx_1 )[0]
		msa_idx_2 = np.where(sp_seq_idx_2 == sp_idx_2 )[0]

	# +1 added to ids because the first element is the seq of interest
	# and not used while calculating species count, index, etc.
		file1_species_msa = [ msa_1[id1+1] for id1 in msa_idx_1 ]
		file2_species_msa = [ msa_2[id2+1] for id2 in msa_idx_2 ]
 
	# the file with less entry for species MSA requires padded entries (to match number of MSAs per species)

	
		if len(msa_idx_1) == len(msa_idx_2):
				pass
		elif len(msa_idx_1) > len(msa_idx_2): 
				pad_num = len(msa_idx_1) - len(msa_idx_2)
				pad_filler = "-" * seq_2_size
				for p in range(pad_num):
						pad_str = "Padding_"+str(p+1)+"|"+sp+"|xxx"
						file2_species_msa.append((pad_str, pad_filler))
		else:
			pad_num = len(msa_idx_2) - len(msa_idx_1)
			pad_filler = "-" * seq_1_size
			for q in range(pad_num):
				pad_str = "Padding_"+str(q+1)+"|"+sp+"|9.999E-27"
				file1_species_msa.append((pad_str, pad_filler))
		assert len(file1_species_msa) == len (file2_species_msa), f" numb of species MSA don't match after padding - {sp} "
		file1_processed_msa_list.extend(file1_species_msa)
		file2_processed_msa_list.extend(file2_species_msa)
	
		# write the processed data as a pickle file
		fname1 = pathlib.Path(file1).stem
		fname2 = pathlib.Path(file2).stem
		fh1 = pathlib.Path(out_dir, fname1+"_processed.fasta")
		fh2 = pathlib.Path(out_dir, fname2+"_processed.fasta")
		save_processed_msa_data (file1_processed_msa_list, fh1)
		save_processed_msa_data (file2_processed_msa_list, fh2)

# saves padded / processed MSA to a pickle file
# the data is in the format needed for DiffPalm

def save_processed_msa_data (data, fpath):
		open_file = open(fpath, "w")
		#pickle.dump(data, open_file)
		for (line1, line2) in data:
			record = SeqRecord(
					 Seq(line2),
					 description=line1,
					 id = line1.split("|")[0]
			)
			SeqIO.write(record, open_file, "fasta")
			#open_file.write(line1+os.linesep+line2+os.linesep)
		open_file.close()

# defines the pattern for the species name in the input file
# this is at present specific to DiffPalm

data_separator = "|"
position_of_species_name = 1
get_species_name = (lambda strn: strn.split(data_separator)[position_of_species_name])


if __name__ == '__main__' :

	parser = ArgumentParser(
		description="Preprocessor for creating balanced MSA files for use by DiffPalm"
	)

	parser.add_argument("files", nargs="+", help="Name(s) of one zip or two fasta files.")
	parser.add_argument("--o", "--outdir", help="Output directory name for processed fasta files")
	parser.add_argument("--f", "--format", choices=['DiffPalm', 'ESMPair', 'MLM'], 
						help="Output format - DiffPalm, DSMPair, etc.")
	args = parser.parse_args()

	in_files = args.files
	#out_dir = args.outdir
	print (in_files)
	out_dir = pathlib.Path.cwd()

	if len(in_files) == 1 and zipfile.is_zipfile (in_files[0]):
	# process zip file
		print ("TO DO - processing for zipped version")

	elif len(in_files) == 2:
		process_two_fasta_files (in_files[0], in_files[1], out_dir) 
	else:
		sys.exit ('Please input the name(s) of a zip or two fasta files for pre-processing.')

