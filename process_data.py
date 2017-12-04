#-*- coding: utf-8 -*-
import glob
import numpy as np

files = []


def main():
    tags = []
    for i in glob.glob('/data2/wangshuo/pkl_dir3/*dists.npy'):
        tag = i[25:27]
        for j in glob.glob('/data2/wangshuo/pkl_dir3/*%scounts*' % tag):
            if tag in tags:
                continue
            else:
                tags.append(tag)
                files.append([i, j])
    print tags
    print len(tags)
    print files
    print len(files)
    for i, val in enumerate(files):
        print i, val
        #if i == 3: break
        if i == 0:
            dist = np.load(val[0]) 
            count = np.load(val[1]) 
        else:
            dist += np.load(val[0]) 
            count += np.load(val[1]) 

    print dist.shape
    print count.shape
    np.save('/data2/wangshuo/dist.npy', dist)
    np.save('/data2/wangshuo/count.npy', count)
            

    
def get_avg_flow():
    counts = np.load('/data2/wangshuo/count.npy')
    dists = np.load('/data2/wangshuo/dist.npy')
    counts = counts.astype(np.float64)
    dists = dists.astype(np.float64)
    eps = np.finfo(np.float64).eps
    counts2 = np.copy(counts)
    counts[np.where(counts==0)] = eps
    dists_avg = np.true_divide(dists, counts)
    dists_avg[np.where(counts2==0)] = 0
    print dists_avg.shape
    np.save('/data2/wangshuo/dist_avg.npy', dists_avg)


def analysis():
    distavg = np.load('/home/shawnwang/workspace/research/flow_network_embedding/data/dist_avg.npy')
    a = np.copy(distavg)
    print a
    print '------------'
    print a.shape
    print '------------'
    b = a.flatten()
    b = b[b.nonzero()]

    c = np.sort(b)
    print c
    #plt.plot(c)
    #plt.show()

    bins = np.linspace(np.ceil(np.min(c)),
                       np.floor(np.max(c)),
                       30)

    plt.hist(c, bins = bins)
    plt.title("histogram")
    plt.show()

    print len(np.where(c==1.0)[0])

    #s = c[10000:100000]
    #plt.bar(np.arange(len(s)),s)
    #plt.show()

if __name__ == '__main__':
    #analysis()
    main()
    get_avg_flow()
