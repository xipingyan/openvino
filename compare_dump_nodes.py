import argparse

def read_fn(fn):
    file = open(fn, "r")
    content = file.read()
    # print(content)
    file.close()
    fdata = [float(idx if str(idx) != '' else 0) for idx in content.split(',')]
    return fdata

def compare_file(data1, data2, thr):
    print("Start compare:")
    if len(data1) != len(data2):
        print("  Size is diff:", len(data1), len(data2))
    else:
        diffs=[]
        for i in range(len(data1)):
            diff = abs(data1[i] - data2[i])
            if (diff > thr):
                print(f"  idx={i}, {data1[i]} vs {data2[i]}, abs(diff)={diff}")
                diffs.append((i, diff, data1[i], data2[i]))
        
        if len(diffs) > 0:  
            print(f"  Diff count/Total={len(diffs)}/{len(data1)} are different.")
    print("Finish.")

def main():
    parser = argparse.ArgumentParser(description='Compare 2 dump node files.')
    parser.add_argument('bin_file_1', type=str,
                        help='Plugin(CPU) execution node dump bin file.')
    parser.add_argument('bin_file_2', type=str,
                        help='Plugin(Template) execution node dump bin file.')

    parser.add_argument('t', default=0.0001, type=float, action='store',
                        help='Threshold, default 1-e4')

    args = parser.parse_args()
    print(f"Compare 2 files: \n    {args.bin_file_1}\n    {args.bin_file_2}\n    threshold={args.t}")

    data1 = read_fn(args.bin_file_1)
    data2 = read_fn(args.bin_file_2)
    compare_file(data1, data2, args.t)

if __name__ == "__main__":
    main()