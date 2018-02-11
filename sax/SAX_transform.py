
# coding: utf-8
#import matplotlib.pyplot as plt
#import matplotlib
import h5py
import numpy as np

import sys,os
from pysax import pysax

import argparse
import pprint
import time



def saxize(data,mask,stride=0,window=0,nbins=0,threshold=0.9,frac=None,
           mlabel='0123',dlabel='12345678',verbose=0,**kwargs):

    print('data.shape',data.shape)
    window = window if window!=0 else data.shape[1]
    stride = stride if  stride!=0 else window
    if frac is None:
        nbins = nbins if nbins != 0 else data.shape[1]//4
    else:
        nbins=int(data.shape[1]*frac)
    
    sax = pysax.SAXModel(window = window,stride = window,nbins = nbins,alphabet=dlabel)
    #msax = pysax.SAXModel(window = window,stride = window,nbins = nbins,alphabet=mlabel)
    
    sax_array = np.empty(shape=[data.shape[0],nbins],dtype=np.float)
    mask_array = np.empty(shape=[mask.shape[0],nbins],dtype=np.float)

    if verbose>0: print('ny,nbins=',data.shape[0],nbins)
    for i in range(data.shape[0]):        
        data_intensity = sax.symbolize_signal(data[i,:],array=True,**kwargs)      
        #print('data_intensity len',len(data_intensity),type(data_intensity),data_intensity)
        sax_array[i,:]=data_intensity        

        #mask_intensity = msax.symbolize_signal(mask[i,:],array=True,**kwargs) 
        mask_intensity = sax.slide_window(mask[i,:])
        if verbose>0 and i%50==0: 
            print('loop=',i,data_intensity[::300],mask_intensity[::300])
        mask_array[i,:] = mask_intensity>threshold

    return sax_array,mask_array,sax.slide_window

def map2I(x):
    return x

def h5_to_saxh5(filename,output,**kwargs):
    with h5py.File(filename,"r") as fp: 
        data = fp['data'].value
        mask = fp['mask'].value
        freq = fp['frequencies'].value
        time = fp['time'].value
        ra = fp['ra'].value
        dec = fp['dec'].value
        ref_channel = fp['ref_channel'].value

    print('output',output)
    sax_data,sax_mask,slide_window = saxize(data,mask,**kwargs)
    #sax_data,sax_mask,slide_window = (data,mask,map2I)
    print('sax_data shape:',sax_data.shape)
    print('sax_mask shape:',sax_mask.shape)
    nbins=sax_data.shape[1]

    with h5py.File(output, "w") as fp:
        fp["data"] = sax_data
        fp["mask"] = sax_mask
        fp["frequencies"] = freq
        fp["time"] = slide_window(time)
        fp["ra"] = slide_window(ra)
        fp["dec"] = slide_window(dec)
        fp["ref_channel"] = ref_channel



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Symbolic Aggregate approXimation (SAX) transformation. \
                                             Assume a gapless (fixed freq. no missing value) time series')
    parser.add_argument('--fin', dest='fin', help='hdf5 filename to input time series data',
                        default=None, type=str)
    parser.add_argument('--fout', dest='fout',
                        help='hdf5 filename to output SAX transformed data',
                        default='', type=str)
    parser.add_argument('--outdir', dest='outdir',
                        help='directory to output SAX transformed data',
                        default='', type=str)    
    parser.add_argument('--window', dest='window',
                        help='sliding window length to define the number of words',
                        default=0, type=int)
    parser.add_argument('--stride', dest='stride',
                        help='stride of sliding, if stride < window, there is overlapping in windows',
                        default=0, type=str)
    parser.add_argument('--nbin', dest='nbin',
                        help='number of bins in each sliding window, defining the length of word',
                        default=0, type=int)
    parser.add_argument('--frac', dest='frac',
                        help='fraction of original data to use, defines the length of word',
                        default=None, type=float)
    parser.add_argument('--threshold', dest='threshold',
                        help='variable if mask<threshold set zero',
                        default=0.5, type=float) 
    parser.add_argument('--parallel', dest='parallel',
                        help='type of parallelization',
                        default=None, type=str)
    parser.add_argument('--msymbol', dest='mask_symbols',
                        help='alphabets for SAXing mask',
                        default='3210', type=str)
    parser.add_argument('--dsymbol', dest='data_symbols',
                        help='alphabet for symbolization, also determines number of value levels',
                        default='12345678', type=str)    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.window==0:
        print('window not passed, setting nwindow=stride=len(data)')
    if args.frac is None and args.nbin==0:
        print('nbin not passed, setting nbin=ndata/4')

    if args.frac is None:
        sax_key='sax_w{}s{}n{}'.format(args.window,args.stride,args.nbin)
    else:
        sax_key='sax{}_w{}s{}'.format(int(args.frac*100),args.window,args.stride)
        
    data_dir=os.path.dirname(args.fin)
    outdir=args.outdir if args.outdir!='' else os.path.join(data_dir,sax_key)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    if '*' in args.fin:
        files = glob.glob(args.fin)
        outfiles = [os.path.join(outdir,sax_key,os.path.basename(x)) for x in files]
    else:
        files=[args.fin]
        outfiles=[os.path.join(outdir,sax_key+'_'+os.path.basename(args.fin))]
        
    print('Called with args:')
    print(args)

    for fin, fout in zip(files, outfiles):
        print('{} - > {}'.format(fin,fout))        
        h5_to_saxh5(fin,fout,
                window=args.window,
                stride=args.stride,
                nbins=args.nbin,
                mlabel=args.mask_symbols,
                dlabel=args.data_symbols,
                frac=args.frac,
                threshold=args.threshold,
                parallel=args.parallel)

