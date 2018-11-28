import json
import click
from preprocessing import read_pickle


def write_to_json(data, output):
	with open(output, 'w') as fp:
        	fp.write(data + '\n')

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_json', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_json):
	df=read_pickle(input_file)
	data= df.to_json(orient='records')
	write_to_json(data,output_json)

if __name__ == '__main__':
    main()	
