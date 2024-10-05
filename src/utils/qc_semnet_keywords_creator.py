"""
Create a list of Quamtum Computing related keywords from SEMNET.

- input file: SEMNET_concepts_updated.csv
- output file: qc_semnet_keywords_updated.txt
- reference file: semnet_keywords_updated.txt (debug purpose: to check if all keywords are correct.)

- Date: 06/10/2024
- Author: Gilchan Park

"""

import os
import csv
import argparse


def generate_qc_keywords(
    data_dir,
    input_file,
    output_file,
):
    qc_related_semnet_concept_file = os.path.join(data_dir, input_file) # qc related concept list from SEMNET (annotated by Paul)
    out_file = os.path.join(data_dir, output_file)

    qc_semnet_keywords_updated = []
    with open(qc_related_semnet_concept_file, "r") as fin:
        reader = csv.reader(fin, delimiter=",")
        next(reader) # skip the first row
        for line in reader:
            kw = line[1]
            kw = kw.strip()
            is_qc_concept = line[2]
            if is_qc_concept == 'y':
                qc_semnet_keywords_updated.append(kw)

    qc_semnet_keywords_updated.sort()

    # [START] - debug
    semnet_keywords_updated_file = os.path.join(data_dir, "semnet_keywords_updated.txt")
    semnet_keywords_updated = []
    with open(semnet_keywords_updated_file, "r") as fin:
        for line in fin.readlines():
            idx, kw = line.split('\t')
            kw = kw.strip()
            semnet_keywords_updated.append(kw)
            
    set1 = set(qc_semnet_keywords_updated)
    set2 = set(semnet_keywords_updated)

    # Check if set1 is a subset of set2
    is_subset = set1.issubset(set2)

    # Print the result
    print(f"Is qc_semnet_keywords_updated a subset of semnet_keywords_updated? {is_subset}")
    print(set1.difference(set2))
    # [END] - debug
    
    with open(out_file, "w+") as fout:
        for idx, kw in enumerate(qc_semnet_keywords_updated):
            fout.write(f'{idx}\t{kw}\n')


def main():
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True, help="qc related concept list from SEMNET")
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    input_file = args.input_file
    output_file = args.output_file

    generate_qc_keywords(
        data_dir,
        input_file,
        output_file,
    )


if __name__ == "__main__":
    main()
