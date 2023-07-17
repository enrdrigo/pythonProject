import numpy as np
from numba import njit

# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE POSITION OF THE OXY AND OF THE TWO HYDROGENS AT GIVEN SNAPSHOT. IT ALSO GETS THE POSITION OF THE
# FOURTH PARTICLE IN THE TIP4P/2005 MODEL OF WATER WHERE THERE IS THE CHARGE OF THE OXY (SEE TIP4P/2005 MODEL OF WATER).

@njit(fastmath=True)
def computeposmol(Np, data_array, posox, natpermol):
    nmol = int(Np / natpermol)
    datamol = np.zeros((8, nmol, natpermol))
    datamol = data_array.reshape((8, nmol, natpermol))

    #
    pos = np.zeros((natpermol, 3, nmol))
    posO = np.zeros((3, nmol))
    posH1 = np.zeros((3, nmol))
    posH2 = np.zeros((3, nmol))
    for i in range(natpermol):
        pos[i][0] = np.transpose(datamol[2])[i]
        pos[i][1] = np.transpose(datamol[3])[i]
        pos[i][2] = np.transpose(datamol[4])[i]
    #


    if natpermol != 1:
        #
        bisdir = np.zeros((3, nmol))
        bisdir = 2 * pos[0] - pos[1] - pos[2]

        #

        #
        poschO = np.zeros((3, nmol))
        poschO = pos[0] - posox * bisdir / np.sqrt(bisdir[0] ** 2 + bisdir[1] ** 2 + bisdir[2] ** 2)

        #
    else:
        poschO = np.zeros((3, nmol))
        poschO = pos[0]

    return poschO, pos


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE CHARGE AND ATOMIC POSITION ARRAYS OF THE ATOMS AT A GIVEN SNAPSHOT. THE OXY POSITION IS SHIFTED ACCORDING
# TO THE TIP4P/2005 MODEL OF WATER.


def computeat(Np, data_array, poschO, pos):
    natpermol = np.shape(pos)[0]
    nmol = int(Np / natpermol)

    #
    chat = np.zeros(Np)
    chat = data_array[5]
    #

    #
    posm = np.zeros((natpermol, nmol, 3))
    posm[0] = np.transpose(poschO)
    for i in range(1, natpermol):
        posm[i] = np.transpose(pos[i])

    #

    #
    pos_at = np.zeros((3, Np))
    test = np.transpose(posm)
    pos_at = test.reshape((3, Np))
    #

    # THIS IS IN FACT THE CHARGE TIMES A PHASE WHERE GAT = 2 * np.pi * np.array((1e-8, 1e-8, 1e-8)) / L. I DO THIS
    # IN ORDER TO COMPUTE PROPERLY THE STATIC DIELECTRIC CONSTANT VIA THE FOURIER TRANFORM OR THE CHARGE OVER
    # THE MODULUS OF G,  AT G \APPROX 0
    ch_at = np.zeros(Np)
    ch_at = chat
    #

    return ch_at, np.transpose(pos_at)

@njit(fastmath=True)
def computeaten(Np, data_array, pos):
    natpermol = np.shape(pos)[0]
    nmol = int(Np / natpermol)
    #
    datamol = np.zeros((8, nmol, natpermol))
    datamol = data_array.reshape((8, nmol, natpermol))
    #
    en0 = np.zeros(nmol)
    enH1 = np.zeros(nmol)
    enH2 = np.zeros(nmol)
    en = np.zeros((3, nmol))
    for i in range(natpermol):
        en[i] = np.transpose(datamol[6])[i] + np.transpose(datamol[7])[i]

    #
    enat = np.zeros(Np)
    enat = data_array[6] + data_array[7]
    #

    #
    pos_at = np.zeros((3, Np))
    pos_at[0] = data_array[2]
    pos_at[1] = data_array[3]
    pos_at[2] = data_array[4]
    #

    #
    en_at = np.zeros(Np)
    en_at = enat
    #

    endip = np.zeros((3, nmol))
    endipn = np.zeros((natpermol, 3, nmol))
    for i in range(natpermol):
        endipn[i] = pos[i]*(en[i]-np.sum(enat)/Np)
    endip = np.sum(endipn, axis=0)

    return en_at, np.transpose(pos_at), np.sum(enat), np.transpose(endip)