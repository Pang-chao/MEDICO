########################################################################
#
# Date: April 2006 Authors: Guillaume Vareille, Michel Sanner
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
# $Header: /mnt/raid/services/cvs/python/packages/share1.5/Vision/PILTypes.py,v 1.4 2006/11/01 21:03:54 vareille Exp $
#
# $Id: PILTypes.py,v 1.4 2006/11/01 21:03:54 vareille Exp $
#

from NetworkEditor.datatypes import AnyArrayType


class ImageType(AnyArrayType):

    from Image import Image
    def __init__(self, name='image', color='#995699', shape='rect',
                 klass=Image):

        AnyArrayType.__init__(self, name=name, color=color, shape=shape, 
                              klass=klass)
