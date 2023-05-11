import datasets
import gc
import argparse
import glob
import os
import transformers
import pandas as pd

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='', help='read and count token for this file')
    parser.add_argument('--folder_path', default='', help='read and count token for all files in this folder')
    parser.add_argument('--file_path_ends_with', default='', help='filter the file names by this suffix')
    parser.add_argument('--chunk_size', default='4096', help='filter the file names by this suffix')
    parser.add_argument('--cpu_count', default='1', help='the number of cpus to run multiprocessing. This speeds up tokenizer.')
    args = parser.parse_args()
    args.chunk_size = int(float(args.chunk_size))
    args.cpu_count = int(args.cpu_count)
    print(f'file_path: {args.file_path}', f'folder_path: {args.folder_path}', f'file_path_ends_with: {args.file_path_ends_with}', f'chunk_size: {args.chunk_size:e}', f'cpu_count: {args.cpu_count}')
    return args

def gather_file_path_list(file_path, folder_path, file_path_ends_with):
    file_path_list = []

    ## handle single file arg
    file_path_list.append(file_path)

    ## handle folder path arg
    for file_path in glob.glob(os.path.join(folder_path, '*.*')):
        file_path_list.append(file_path)

    ## ensure no file overlap between file path and folder path
    assert len(set(file_path_list)) == len(file_path_list), 'duplicated file paths found, maybe because file_path and files in folder_path have overlap'

    ## filter by file path suffix
    filtered_file_path_list = [file_path for file_path in file_path_list if file_path.endswith(file_path_ends_with)]

    return filtered_file_path_list

def tokenize_many_file(file_path_list, chunk_size, cpu_count):
    tokenizer = get_tokenizer()
    df_list = []
    assert len(file_path_list) > 0, 'No file path satisfies'
    for file_path in file_path_list:
        one_file_df = tokenize_one_file(file_path, tokenizer, chunk_size, cpu_count)
        df_list.append(one_file_df)
    df = pd.concat(df_list)
    return df

def get_tokenizer():
    # tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer = transformers.LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')  # use fast not availabel
    return tokenizer

def read_in_chunks(f, chunk_size=1024):
    while True:
        data = f.read(chunk_size)
        if not data:
            break
        yield data

def tokenize_one_file(file_path, tokenizer, chunk_size, cpu_count):
    chunk_index, word_count, token_count = 0, 0, 0
    with open(file_path, 'r') as f:
        for text_chunk in read_in_chunks(f=f, chunk_size=chunk_size):
            word_count += len(text_chunk.split(' '))

            def text_chunk_to_text_piece(text_chunk, piece_size=2000):  # one big text into list of smaller text
                text_piece_list = [text_chunk[i:i + piece_size] for i in range(0, len(text_chunk), piece_size)]
                return text_piece_list

            text_piece_list = text_chunk_to_text_piece(text_chunk)
            text_piece_dataset = datasets.Dataset.from_dict({'text_piece': text_piece_list})

            def count_token(dataset_batch):
                processed_examples = {'text_piece_token_count': [len(text_piece_input_ids) for text_piece_input_ids in tokenizer(dataset_batch['text_piece'])['input_ids']]}
                return processed_examples

            token_count_dataset = text_piece_dataset.map(count_token, batched=True, num_proc=cpu_count, load_from_cache_file=False, batch_size=1000)
            token_count += sum(token_count_dataset['text_piece_token_count'])

            print(f'chunk {chunk_index} completed')
            chunk_index += 1
            # del text_chunk
            # gc.collect()
    file_size = round(os.path.getsize(file_path) / (1024 ** 2), 10)
    one_file_df = pd.DataFrame({'file_path': [file_path], 'word_count': [word_count], 'token_count': [token_count], 'file_size(MB)': [file_size]})
    return one_file_df

if __name__ == '__main__':
    args = parse_argument()
    file_path_list = gather_file_path_list(file_path=args.file_path, folder_path=args.folder_path, file_path_ends_with=args.file_path_ends_with)
    df = tokenize_many_file(file_path_list=file_path_list, chunk_size=args.chunk_size, cpu_count=args.cpu_count)
    df.to_csv('token_count.csv', index=False)
    df.sum().to_frame().transpose().drop(columns='file_path').to_csv('token_count_sum.csv', index=False)
    print(df)
    print(df.sum())



