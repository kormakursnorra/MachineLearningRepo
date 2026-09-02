from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')

print(newsgroups.shape())