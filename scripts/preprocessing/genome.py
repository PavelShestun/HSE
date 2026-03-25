import os
import urllib.request
from Bio import SeqIO

def download_genome_if_needed(filepath):
    if not os.path.exists(filepath):
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=U00096.3&rettype=fasta&retmode=text"
        urllib.request.urlretrieve(url, filepath)

def load_genome_sequence(filepath):
    with open(filepath, "r") as handle:
        record = next(SeqIO.parse(handle, "fasta"))
    return str(record.seq).upper()