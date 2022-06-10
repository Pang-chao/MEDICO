########################################################################
#
# Date: November 2006 Authors: Guillaume Vareille, Michel Sanner
#
#    sanner@scripps.edu
#    vareille@scripps.edu
#
#       The Scripps Research Institute (TSRI)
#       Molecular Graphics Lab
#       La Jolla, CA 92037, USA
#
# Copyright: Guillaume Vareille, Michel Sanner and TSRI
#
# revision:
#
#########################################################################
#
# $Header: /mnt/raid/services/cvs/python/packages/share1.5/Vision/matplotlibTypes.py,v 1.2 2006/11/01 22:18:00 vareille Exp $
#
# $Id: matplotlibTypes.py,v 1.2 2006/11/01 22:18:00 vareille Exp $
#

from NetworkEditor.datatypes import AnyArrayType

class MPLFigureType(AnyArrayType):
    from matplotlibNodes import Figure
    def __init__(self, name='MPLFigure', color='#99AFD8', shape='oval',
                 klass=Figure):
      
        AnyArrayType.__init__(self, name=name, color=color, shape=shape, 
                              klass=klass)



class MPLAxesType(AnyArrayType):
    from matplotlib.axes import Axes
    def __init__(self, name='MPLAxes', color='#99AFD8', shape='oval',
                 klass=Axes):
      
        AnyArrayType.__init__(self, name=name, color=color, shape=shape, 
                              klass=klass)



class MPLDrawAreaType(AnyArrayType):

    def __init__(self, name='MPLDrawArea', color='#99AFD8', shape='diamond',
                 klass=dict):
      
        AnyArrayType.__init__(self, name=name, color=color, shape=shape, 
                              klass=klass)
