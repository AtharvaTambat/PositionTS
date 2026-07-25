def read_hess(filename, num_atoms):
    import numpy as np
    import re
    read=False
    lines =[]
    for line in open(filename):
        if('Hessian of the SCF Energy' in line):
            read=True
            continue
        elif(read and 'Gradient time:  CPU' in line):
            break
        if(read):
            lines.append(line.strip())

    list_words = list( map( lambda line: re.split('\s+', line ), lines) )
    results = np.zeros((3*num_atoms, 3*num_atoms) )
    indices = np.array( list_words[0], dtype=int)
    n_words = len(indices)+1

    for words in list_words:

        if(len(words)!=n_words):
            indices = np.array( words, dtype=int)
            n_words = len(indices)+1
            continue

        i_atoms = int( words[0] )
        results[i_atoms-1, indices-1] = np.array(words[1:], dtype=float)

    return results
def read_entropy(filename):
    import re
    for line in reversed(open(filename).readlines()):
        if ("Vibrational Entropy:" in line):
            s_vib= float(re.split("\s+", line)[3])
        elif("Rotational Entropy:" in line):
            s_rot = float(re.split("\s+", line)[3])
            break
    return (s_vib, s_rot)
if __name__=='__main__'    :
    import numpy as np
    from ard_gsm.qchem import QChem
    from ase.data import atomic_masses

    i = 0
    directory=  'wb97xd3/'
    symbols, pos = QChem(logfile=directory+"/rxn"+str(i).zfill(6)+"/"+case+str(i).zfill(6)+".log").get_geometry()
    mass
    print(read_hess('wb97xd3/rxn000000/p000000.log', 9) )
    evals, evecs = np.linalg.eig( read_log('wb97xd3/rxn000000/p000000.log', 9) )
    indices = np.argsort(evals)
    evals = evals[indices]
    evecs = evecs[indices]
    print(evals)