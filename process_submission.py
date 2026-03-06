from pathlib import Path

def process(partition_path, output_path):

    clusters = []
    with open(partition_path) as f:
        lines = f.readlines()
    this_cluster = []
    for line in lines:
        if len(line.split(" ")) == 2:
            if this_cluster:
                clusters.append(this_cluster)
                this_cluster = []
        elif len(line.split(" ")) == 3:
            filename_with_start, start_time, end_time = line.strip().split()
            this_cluster.append((filename_with_start, start_time, end_time))
        
    clusters.append(this_cluster)

    with open(output_path, "w") as f:
        for id, cluster in enumerate(clusters):
            f.write(f"Class {id}\n")
            for filename_with_start, start_time, end_time in cluster:
                filename_without_start = filename_with_start.split("-")[0]
                segment_start = float(filename_with_start.split("-")[1])
                start = round(float(start_time) + segment_start, 4)
                end = round(float(end_time) + segment_start, 4)
                f.write(f"{filename_without_start} {start} {end}\n")
            f.write("\n")

if __name__  ==  "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Zerospeech data by slicing audio and creating TextGrid files based on phoneme alignments.")
    parser.add_argument("partition_path", type=Path, help="Output partition path.")
    parser.add_argument("output_path", type=Path, help="Submission output path.")

    args = parser.parse_args()

    process(args.partition_path, args.output_path)