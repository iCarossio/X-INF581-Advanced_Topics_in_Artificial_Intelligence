from scipy.io import loadmat
import numpy as np
from scipy.sparse.linalg import eigs


############## Question 1
# Implement Algorithm 1

def generate_spectral_embeddings(A, d):
	# Function that generates spectral embeddings
	
	##################
	# your code here #
	##################

	return U


def write_to_disk(U):
	fout = open('embeddings/spectral_embeddings', 'w', encoding="UTF-8")
	for i in range(U.shape[0]):
		e = U[i,:]
		e = ' '.join(map(lambda x: str(x), e))
		fout.write('%s %s\n' % (i, e))


if __name__ == "__main__":
	d = dict()
	loadmat('data/Homo_sapiens.mat', mdict=d)
	A = d['network']
	U = generate_spectral_embeddings(A, 128)
	write_to_disk(U)
