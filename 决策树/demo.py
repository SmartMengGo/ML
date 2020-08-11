import tree
import treePlotter

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
labels=['age','prescript','astigmatic','tearRate']
tree=tree.createTree(lenses,labels)
print(tree)