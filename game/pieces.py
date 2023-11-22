import cocos
import pyglet
import os
import sys
DIR = os.path.dirname(__file__)
sys.path.append(DIR)
from utils import PieceType, ChessImages, CircleLayer

class Piece(cocos.layer.Layer):
    def __init__(self,piece_type, side, coord,radius = 40, **kwargs):
        super().__init__()
        image = ChessImages.get_piece_image(piece_type,side)
        self.image = cocos.sprite.Sprite(image)
        w, h = self.image.width, self.image.height
        self.image.scale = radius/w
        self.side = side
        self.piece_type = piece_type
        self.is_selected = False
        self.legal_moves = list()
        self.coord = coord
        self.add(self.image)


    
