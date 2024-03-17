import numpy as np
from mpi4py import MPI
def distBox(slab,myid,lx,ly,lz,npx,nprocs,mpi_comm):

    # Section #4 augment slabs with boundary regions

#    print('in',myid,slab.shape)        

    plane = np.zeros((ly,lz),dtype='float64')    
    topslice=slab[npx-1]
    botslice=slab[0]
    
    if (nprocs%2 == 0) or (myid != nprocs-1) :
        if (myid % 2) == 0:
            #    file1.write("send \n")
            target=myid+1
            mpi_comm.Send([topslice,MPI.DOUBLE],dest=target,tag=77)
            
        else:
            #    file1.write("recv \n")
            target=myid-1
            mpi_comm.Recv([plane,MPI.DOUBLE],source=target,tag=77)
            
            
    #file1.write("top slice 2 \n")
    if ((myid != 0) and ((myid != nprocs-1) or (nprocs%2!=0))):
        if (myid % 2) == 0:
            #        file1.write("recv \n")
            target=myid-1
            mpi_comm.Recv([plane,MPI.DOUBLE],source=target,tag=67)
        else:
            #        file1.write("send \n")
            target=myid+1
            mpi_comm.Send([topslice,MPI.DOUBLE],dest=target,tag=67)
            

    slab=np.append(plane,slab).reshape(npx+1,ly,lz)
    plane = np.zeros((ly,lz),dtype='float64')    

    #file1.write("bot slice 1 \n")

    if (nprocs%2 == 0) or (myid != nprocs-1) :
        if (myid % 2) == 0:
            #    file1.write("recv \n")
            target=myid+1    
            mpi_comm.Recv([plane,MPI.DOUBLE],source=target,tag=57)
        else:
        #    file1.write("send \n")
            target=myid-1
            mpi_comm.Send([botslice,MPI.DOUBLE],dest=target,tag=57)


    #file1.write("bot slice 2 \n")
    if ((myid != 0) and ((myid != nprocs-1) or (nprocs%2!=0))):
        if (myid % 2) == 0:
            #        file1.write("send \n")
            target=myid-1
            mpi_comm.Send([botslice,MPI.DOUBLE],dest=target,tag=47)
        else:
            #        file1.write("recv \n")
            target=myid+1    
            mpi_comm.Recv([plane,MPI.DOUBLE],source=target,tag=47)
    
    slab=np.append(slab,plane).reshape(npx+2,ly,lz)
#    print("slab shape1",slab.shape)
    return(slab)


