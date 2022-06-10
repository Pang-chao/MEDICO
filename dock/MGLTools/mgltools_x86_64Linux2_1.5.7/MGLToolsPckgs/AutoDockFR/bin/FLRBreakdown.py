import numpy

def FRLBreakdown(scorer):
################  FRL  ####################################################

    FE_coeff_vdW_42		= 0.1662 # van der waals
    FE_coeff_hbond_42	= 0.1209 # hydrogen bonding
    FE_coeff_estat_42	= 0.1406 # electrostatics
    FE_coeff_desolv_42	= 0.1322 # desolvation
    FE_coeff_tors_42	= 0.2983 # torsional 

    #scorer = AD42ScoreC
    terms = scorer.FRLmolSyst.scorer.get_terms()
    names = scorer.FRLmolSyst.scorer.names
    #import numpy
    hArray = numpy.array(terms[1].get_score_array(), 'f')*FE_coeff_hbond_42
    eArray = numpy.array(terms[0].get_score_array(), 'f')*FE_coeff_estat_42
    vdwArray = numpy.array(terms[2].get_score_array(), 'f')*FE_coeff_vdW_42
    dsArray = numpy.array(terms[3].get_score_array(), 'f')*FE_coeff_desolv_42

##     #hArray *= FE_coeff_hbond_42
##     #eArray *= FE_coeff_estat_42
##     #vdwArray *= FE_coeff_vdW_42
##     #dsArray *= FE_coeff_desolv_42


    distance = scorer.FRLmolSyst.get_distance_matrix(1, 0)

    ct2=1
    perResLBreakDown={}
    flexRecAtoms = scorer.flexRecAtoms
    interfaceAtoms = scorer.interfaceAtoms
    ligAtoms = scorer.ligAtoms

    sc = flexRecAtoms+interfaceAtoms
 #   reslenct = 0
 #   for resl in reslen:
        
## #        for i in range(reslenct-1,resl):
    for i in range(len(sc)):
        #reslenct += 1
        esum = 0.
        vsum = 0.
        hsum = 0.
        dsum = 0.
        for j in range(len(ligAtoms)):
            e  = eArray[i][j]
            v = vdwArray[i][j]
            h = hArray[i][j]
            d = dsArray[i][j]
            if i == 33:
                if h != 0.0:
                    print j, h
            esum += e
            vsum += v
            hsum += h
            dsum += d
        #if reslenct in reslen:receptorFT.flexNodeList[ct2-1].name'Res':flexRecAtoms[ct2-1].autodock_element
        perResLBreakDown.update({ct2:{'Res':"%8s-%4s"%(sc[ct2-1].parent.name,sc[ct2-1].name),'elec':esum, 'hbonds':hsum, 'vdw':vsum, 'ds':dsum}})
        ct2+=1

    FRLPerAtm =[[0,'-',0.0,0.0,0.0,0.0] for j in range(len(sc))]
    FRLPerAtmSum =[]
    elec2=0.
    vdw2=0.
    hbond2=0.
    ds2=0.
    for k,v in perResLBreakDown.items():
        n = k-1
        FRLPerAtm[n][0] = k
        for k1,v1 in v.items():
            if k1 == 'Res':
                FRLPerAtm[n][1] = v1
            elif k1=='vdw':
                FRLPerAtm[n][2] = v1
            elif k1=='hbonds':
                FRLPerAtm[n][3] = v1
            elif k1=='elec':
                FRLPerAtm[n][4] = v1
            elif k1=='ds':
                FRLPerAtm[n][5] = v1
 


        
    ## print "             FRFR per-Atom Intramolecular Energy Analysis"
    ## print "             =============================================="
    ## print
    ## print ("%15s %5s %9s %9s %9s"%('Num','Type','elec','hbonds','vdw'))
    #termsName = ['Type','elec','hbonds','vdw','dsum']
    elec2=0.
    vdw2=0.
    hbond2=0.
    ds2=0.
    for i in range(1,ct2):
        #print ("%15d %5s %9.4f %9.4f %9.4f"%(i,perResBreakDown[i]['Res'],perResBreakDown[i]['elec'],perResBreakDown[i]['hbonds'],perResBreakDown[i]['vdw']))
        elec2 += perResLBreakDown[i]['elec']
        hbond2 += perResLBreakDown[i]['hbonds']
        vdw2 += perResLBreakDown[i]['vdw']
        ds2 += perResLBreakDown[i]['ds']
    #print"---------------------------------------------------------------------------------------------------------"
    #print "%12s %8.4f %9.4f %9.4f %9.4f\n\n\n"%("Sum:",elec2+hbond2+vdw2,elec2,hbond2,vdw2)
    
    FRLPerAtmSum =["Sum:",elec2+hbond2+vdw2+ds2,vdw2,hbond2,elec2,ds2]


     #############################################################################

    print "                          FRL            Energy Analysis            |        FR - per-Atom Intermolecular Energy Analysis"
    print "            ==============================================          |        =============================================="
    print
    print ("%12s %12s %9s %9s %9s %13s %12s %12s %9s %9s %9s"%('Num','Residue','vdw','hbonds','elec',"|",'Num','Residue','vdw','elec','ds'))
    merge = FRLPerAtm
    for i in range(len(FRLPerAtm)):
        merge[i].append("|")

    for lPA in merge:
        for t in lPA:
            if isinstance(t,float):
                print '%9.4f'%t,
            else:
                print '%12s'%t,

        print "|"
    print "=================================================================================================================================="
    merge2 = FRLPerAtmSum
    merge2.append("|")
    cnt = 0
    for t in merge2:
        if isinstance(t,float):
            print '%9.4f'%t,
        else:
            cnt+=1
            if cnt in [1,3]:
                print '%16s'%t,
            else:
                print '%12s'%t,
            

    print
    print
    print

#########################################################################################   

#FRLBreakdown(pop[0].scorer)
FRLBreakdown(self.scorer)
#execfile("../AutoDockFR/bin/FLRBreakdown.py")
