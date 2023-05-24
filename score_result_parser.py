import pandas as pd


def main():
    df = pd.read_csv('result/scores.txt', sep='\t')
    df['mean'] = df['mean'].apply(lambda x: x * 100)
    print(''.join([f"({row['episodes'] / 1000:.0f},{row['mean']})" for index, row in df.iterrows()]))
    print('\n'.join([f"{row['episodes']:.0f} & {row['mean']}" for index, row in df.iterrows()]))


if __name__ == '__main__':
    main()
