import numpy as np
rng=np.random.default_rng(809);n=256;m=62
x=rng.normal(size=n)+1j*rng.normal(size=n);h=(rng.normal(size=m)+1j*rng.normal(size=m))/np.sqrt(m)
idx=np.pad(np.arange(n),(m-1,0),mode='reflect');columns=idx[np.arange(n)[:,None]+m-1-np.arange(m)[None,:]]
H=np.zeros((n,n),dtype=np.complex128);np.add.at(H,(np.arange(n)[:,None],columns),np.broadcast_to(h,(n,m)))
prefix=np.pad(x,(m-1,0),mode='reflect')[:m-1];expected=np.convolve(np.concatenate((prefix,x)),h,mode='full')[m-1:m-1+n]
assert np.allclose(H@x,expected,rtol=1e-11,atol=1e-11)
y=H@x;eig,V=np.linalg.eigh(H.conj().T@H);rhs=V.conj().T@(H.conj().T@y);pv=V.conj().T@x;reg=.1
recovered=V@((rhs+reg*pv)/(np.maximum(eig,0)+reg));assert np.allclose(recovered,x,rtol=1e-9,atol=1e-9)
print('VERIFIED reflected operator and prior-consistent inverse')
