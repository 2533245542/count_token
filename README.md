# count_token
Use a huggingface tokenizer to count the number of tokens in a file.

Using this script to tokenize MIMIC-III NOTEEVETNS.csv table takes around 20 minutes.

## steps
1. Install packages with
`pip install datasets, transformers, pandas`

2. Run the `count_token.py` with specifyinig either 1. a file 2. a folder 3. a folder with a file pattern

`chunk_size` is limited by memory size. When a file is too large like MIMIC's NOTEEVENTS.csv, you may want to read it in chunks instead of one all together to avoid running out of memory. It is found that `chunk_size=3e8` works fine for a machine with 30 GB memory.

`cpu_count` uses multiple CPUs for tokenization to make it faster.


`python count_token.py --file_path '/home/2533245542/temp/NOTEEVENTS.csv' --chunk_size 3e8 --cpu_count 10`

`python count_token.py --folder_path '/home/2533245542/temp/' --chunk_size 3e8 --cpu_count 10`

`python count_token.py --folder_path '/home/2533245542/temp/' --file_path_ends_with '.csv' --chunk_size 3e8 --cpu_count 10`

3. After the script runs to the end, it will save 2 files called `token_count.csv` and `token_count_sum.csv`. In the first file, each line is file name and token count. The meaning of the second file is when you are processing a folder. It will do a sum of the first file to get the total token count. 
