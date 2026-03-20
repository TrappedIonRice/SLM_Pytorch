# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import zbf_zemax as zbf

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def B_spline_new():
    def B_func(i,k,r,t_i,B_list1,r_list):
        if i>30 or k>7:
            return 0
        else:
            if k>1:
                return ((r_list[r]-t_i[i])/(t_i[i+k-1]-t_i[i]))*B_func(i,k-1,r,t_i,B_list1,r_list) + ((t_i[i+k]-r_list[r])/(t_i[i+k]-t_i[i+1]))*B_func(i+1,k-1,r,t_i,B_list1,r_list)
            if k==1:
                return B_list1[i,1,r]

    k_num=6
    n_num=30
    r_max=40
    i_list=np.arange(0,n_num+k_num,1)
    t_list=np.zeros(n_num+k_num)
    r_list=np.arange(0,r_max,0.1)


    B_list=np.zeros((n_num+k_num,k_num,r_list.shape[0]))
    B_list1 = np.zeros((n_num+k_num, k_num ,r_list.shape[0]))
    B_line=np.zeros((k_num,r_list.shape[0]))

    k=1
    t_list[:k]=0
    t_list[n_num:]=r_max
    temp_list=np.arange(0,n_num-k,1)
    print(temp_list)
    t_list[k:n_num]=np.exp(np.log(r_max+1)*temp_list/(n_num-k))-1
    #plt.plot(t_list)
    plt.plot(r_list)
    plt.show()

    for ii in range(0,n_num+k):
        for r in range(0,r_list.shape[0],1):
            if t_list[ii]<=r_list[r] and r_list[r]<t_list[ii+1] :
                B_list[ii,1,r]=1
                B_list1[ii, 1,r] = 1

    plt.imshow(B_list[:, 1, :])
    plt.title(1)
    plt.colorbar()
    plt.show()
    for kk in range(2,k_num):
        for ii in range(0,t_list.shape[0],1):
            print(ii)
            for r in range(0,r_list.shape[0],1):
                B_list[ii,kk,r]=B_func(ii,kk,r,t_list,B_list1,r_list)

        plt.imshow(B_list[:,kk,:])
        plt.title(kk)
        plt.colorbar()
        plt.show()
        for iix in range(0,n_num):
            plt.plot(B_list[iix,kk,:])
        plt.show()


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print_hi('PyCharm')
    #B_spline()
    #B_spline_new()
    zbf.zbf_func()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
