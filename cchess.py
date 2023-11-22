# This file is part of the python-chess library.
# Copyright (C) 2012-2021 Niklas Fiekas <niklas.fiekas@backscattering.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
A chess library with move generation and validation,
Polyglot opening book probing, PGN reading and writing,
Gaviota tablebase probing,
Syzygy tablebase probing, and XBoard/UCI engine communication.
"""

from __future__ import annotations

__author__ = "Niklas Fiekas"

__email__ = "niklas.fiekas@backscattering.de"

__version__ = "1.10.0"

import collections
import copy
import dataclasses
import enum
import math
import re
import itertools
import typing
from termcolor import colored

from typing import ClassVar, Callable, Counter, Dict, Generic, Hashable, Iterable, Iterator, List, Mapping, Optional, SupportsInt, Tuple, Type, TypeVar, Union

try:
    from typing import Literal
    _EnPassantSpec = Literal["legal", "fen", "xfen"]
except ImportError:
    # Before Python 3.8.
    _EnPassantSpec = str  # type: ignore


Color = bool
COLORS = [RED, BLACK] = [True, False]
COLOR_NAMES = ["red", "black"]

PieceType = int
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
               SOLDIER, CANNON, FLAG, CHARIOT, HORSE, ELEPHANT, ADVISOR, GENERAL] = range(1, 15)

PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k", 
                 "s", "c", "f", "x", 'h', 'e', 'a', 'g']
PIECE_NAMES = [None, "pawn", "knight", "bishop", "rook", "queen", "king",
               "soldier", "cannon", "flag", "chariot", "horse", "elephant", "advisor","general"]

def piece_symbol(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_SYMBOLS[piece_type])

def piece_name(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_NAMES[piece_type])

UNICODE_PIECE_SYMBOLS = {
    "r": "♖ ", "R": "♜ ",
    "n": "♘ ", "N": "♞ ",
    "b": "♗ ", "B": "♝ ",
    "q": "♕ ", "Q": "♛ ",
    "k": "♔ ", "K": "♚ ",
    "p": "♙ ", "P": "♟ ",
    "s": "卒", "S": "兵",
    "c": "砲", "C": "炮",
    "f": "旗", "F": "火",
    "x": "車", "X": "俥",
    "h": "馬", "H": "傌",
    "e": "象", "E": "相",
    "a": "士", "A": "仕",
    "g": "將", "G": "帥",
}

for sym, unicode  in UNICODE_PIECE_SYMBOLS.items():
    if sym.islower():
        UNICODE_PIECE_SYMBOLS[sym] = colored(unicode,'blue')
    else:
        UNICODE_PIECE_SYMBOLS[sym] = colored(unicode,'red')



FILE_NAMES = [str(i) for i in range(16)]

RANK_NAMES = ["0","1", "2", "3", "4", "5", "6", "7", "8","9"]

# STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STARTING_FEN = "XHEAGAEHX/9/1C1F1F1C1/S1S1S1S1S/9/9/9/9/ppppppppp/rnbqkqbnr b 1"
"""The FEN for the standard chess starting position."""

STARTING_BOARD_FEN = "XHEAGAEHX/9/1C1F1F1C1/S1S1S1S1S/9/9/9/9/ppppppppp/rnbqkqbnr"
"""The board part of the FEN for the standard chess starting position."""


class Status(enum.IntFlag):
    VALID = 0
    NO_WHITE_KING = 1 << 0
    NO_BLACK_KING = 1 << 1
    TOO_MANY_KINGS = 1 << 2
    TOO_MANY_WHITE_PAWNS = 1 << 3
    TOO_MANY_BLACK_PAWNS = 1 << 4
    PAWNS_ON_BACKRANK = 1 << 5
    TOO_MANY_WHITE_PIECES = 1 << 6
    TOO_MANY_BLACK_PIECES = 1 << 7
    BAD_CASTLING_RIGHTS = 1 << 8
    INVALID_EP_SQUARE = 1 << 9
    OPPOSITE_CHECK = 1 << 10
    EMPTY = 1 << 11
    RACE_CHECK = 1 << 12
    RACE_OVER = 1 << 13
    RACE_MATERIAL = 1 << 14
    TOO_MANY_CHECKERS = 1 << 15
    IMPOSSIBLE_CHECK = 1 << 16

STATUS_VALID = Status.VALID
STATUS_NO_WHITE_KING = Status.NO_WHITE_KING
STATUS_NO_BLACK_KING = Status.NO_BLACK_KING
STATUS_TOO_MANY_KINGS = Status.TOO_MANY_KINGS
STATUS_TOO_MANY_WHITE_PAWNS = Status.TOO_MANY_WHITE_PAWNS
STATUS_TOO_MANY_BLACK_PAWNS = Status.TOO_MANY_BLACK_PAWNS
STATUS_PAWNS_ON_BACKRANK = Status.PAWNS_ON_BACKRANK
STATUS_TOO_MANY_WHITE_PIECES = Status.TOO_MANY_WHITE_PIECES
STATUS_TOO_MANY_BLACK_PIECES = Status.TOO_MANY_BLACK_PIECES
STATUS_BAD_CASTLING_RIGHTS = Status.BAD_CASTLING_RIGHTS
STATUS_INVALID_EP_SQUARE = Status.INVALID_EP_SQUARE
STATUS_OPPOSITE_CHECK = Status.OPPOSITE_CHECK
STATUS_EMPTY = Status.EMPTY
STATUS_RACE_CHECK = Status.RACE_CHECK
STATUS_RACE_OVER = Status.RACE_OVER
STATUS_RACE_MATERIAL = Status.RACE_MATERIAL
STATUS_TOO_MANY_CHECKERS = Status.TOO_MANY_CHECKERS
STATUS_IMPOSSIBLE_CHECK = Status.IMPOSSIBLE_CHECK


class Termination(enum.Enum):
    """Enum with reasons for a game to be over."""

    CHECKMATE = enum.auto()
    """See :func:`chess.Board.is_checkmate()`."""
    STALEMATE = enum.auto()
    """See :func:`chess.Board.is_stalemate()`."""
    INSUFFICIENT_MATERIAL = enum.auto()
    """See :func:`chess.Board.is_insufficient_material()`."""
    SEVENTYFIVE_MOVES = enum.auto()
    """See :func:`chess.Board.is_seventyfive_moves()`."""
    FIVEFOLD_REPETITION = enum.auto()
    """See :func:`chess.Board.is_fivefold_repetition()`."""
    FIFTY_MOVES = enum.auto()
    """See :func:`chess.Board.can_claim_fifty_moves()`."""
    THREEFOLD_REPETITION = enum.auto()
    """See :func:`chess.Board.can_claim_threefold_repetition()`."""
    VARIANT_WIN = enum.auto()
    """See :func:`chess.Board.is_variant_win()`."""
    VARIANT_LOSS = enum.auto()
    """See :func:`chess.Board.is_variant_loss()`."""
    VARIANT_DRAW = enum.auto()
    """See :func:`chess.Board.is_variant_draw()`."""

@dataclasses.dataclass
class Outcome:
    """
    Information about the outcome of an ended game, usually obtained from
    :func:`chess.Board.outcome()`.
    """

    termination: Termination
    """The reason for the game to have ended."""

    winner: Optional[Color]
    """The winning color or ``None`` if drawn."""

    def result(self) -> str:
        """Returns ``1-0``, ``0-1`` or ``1/2-1/2``."""
        return "1/2-1/2" if self.winner is None else ("1-0" if self.winner else "0-1")


class InvalidMoveError(ValueError):
    """Raised when move notation is not syntactically valid"""


class IllegalMoveError(ValueError):
    """Raised when the attempted move is illegal in the current position"""


class AmbiguousMoveError(ValueError):
    """Raised when the attempted move is ambiguous in the current position"""


Square = int
SQUARES = [
    C00, C01, C02, C03, C04, C05, C06, C07, C08,   C09, C010, C011, C012, C013, C014, C015, 
    C10, C11, C12, C13, C14, C15, C16, C17, C18,   C19, C110, C111, C112, C113, C114, C115, 
    C20, C21, C22, C23, C24, C25, C26, C27, C28,   C29, C210, C211, C212, C213, C214, C215, 
    C30, C31, C32, C33, C34, C35, C36, C37, C38,   C39, C310, C311, C312, C313, C314, C315, 
    C40, C41, C42, C43, C44, C45, C46, C47, C48,   C49, C410, C411, C412, C413, C414, C415, 
    C50, C51, C52, C53, C54, C55, C56, C57, C58,   C59, C510, C511, C512, C513, C514, C515, 
    C60, C61, C62, C63, C64, C65, C66, C67, C68,   C69, C610, C611, C612, C613, C614, C615, 
    C70, C71, C72, C73, C74, C75, C76, C77, C78,   C79, C710, C711, C712, C713, C714, C715, 
    C80, C81, C82, C83, C84, C85, C86, C87, C88,   C89, C810, C811, C812, C813, C814, C815, 
    C90, C91, C92, C93, C94, C95, C96, C97, C98,   C99, C910, C911, C912, C913, C914, C915, 
] = range(16*10)

SQUARE_NAMES = [f"C{r}{f}" for r in RANK_NAMES for f in FILE_NAMES]

def parse_square(name: str) -> Square:
    """
    Gets the square index for the given square *name*
    (e.g., ``a1`` returns ``0``).

    :raises: :exc:`ValueError` if the square name is invalid.
    """
    return SQUARE_NAMES.index(name)

def square_name(square: Square) -> str:
    """Gets the name of the square, like ``a3``."""
    return SQUARE_NAMES[square]

def square(rank_index: int, file_index: int) -> Square:
    """Gets a square number by file and rank index."""
    return (rank_index << 4) + file_index

def square_file(square: Square) -> int:
    """Gets the file index of the square where ``0`` is the a-file."""
    return square & 15

def square_rank(square: Square) -> int:
    """Gets the rank index of the square where ``0`` is the first rank."""
    return square >> 4

def square_distance(a: Square, b: Square) -> int:
    """
    Gets the Chebyshev distance (i.e., the number of king steps) from square *a* to *b*.
    """
    return max(abs(square_file(a) - square_file(b)), abs(square_rank(a) - square_rank(b)))

def square_manhattan_distance(a: Square, b: Square) -> int:
    """
    Gets the Manhattan/Taxicab distance (i.e., the number of orthogonal king steps) from square *a* to *b*.
    """
    return abs(square_file(a) - square_file(b)) + abs(square_rank(a) - square_rank(b))

def square_knight_distance(a: Square, b: Square) -> int:
    """
    Gets the Knight distance (i.e., the number of knight moves) from square *a* to *b*.
    """
    dx = abs(square_file(a) - square_file(b))
    dy = abs(square_rank(a) - square_rank(b))

    if dx + dy == 1:
        return 3
    elif dx == dy == 2:
        return 4
    elif dx == dy == 1:
        if BB_SQUARES[a] & BB_CORNERS or BB_SQUARES[b] & BB_CORNERS: # Special case only for corner squares
            return 4

    m = math.ceil(max(dx / 2, dy / 2, (dx + dy) / 3))
    return m + ((m + dx + dy) % 2)

# def square_mirror(square: Square) -> Square:
#     """Mirrors the square vertically."""
#     return square ^ 0x38

def square_mirror(square: int) -> int:
    """Mirrors the square vertically for a 10x16 board."""
    # In a 10x16 board, each rank (row) has 16 squares, and there are 10 ranks.
    # To mirror vertically, we need to calculate the new square's index.
    rank = square >> 4
    file = square & 15
    new_rank = 9 - rank  # Mirror the rank vertically
    return new_rank << 4 + file

SQUARES_180 = [square_mirror(sq) for sq in SQUARES]


Bitboard = int
BB_EMPTY = 0
BB_ALL = 0x01ff_01ff_01ff_01ff_01ff_01ff_01ff_01ff_01ff_01ff

BB_SQUARES = [
    BB_00, BB_01, BB_02, BB_03, BB_04, BB_05, BB_06, BB_07, BB_08,   _, _, _, _, _, _, _, 
    BB_10, BB_11, BB_12, BB_13, BB_14, BB_15, BB_16, BB_17, BB_18,   _, _, _, _, _, _, _, 
    BB_20, BB_21, BB_22, BB_23, BB_24, BB_25, BB_26, BB_27, BB_28,   _, _, _, _, _, _, _, 
    BB_30, BB_31, BB_32, BB_33, BB_34, BB_35, BB_36, BB_37, BB_38,   _, _, _, _, _, _, _, 
    BB_40, BB_41, BB_42, BB_43, BB_44, BB_45, BB_46, BB_47, BB_48,   _, _, _, _, _, _, _, 
    BB_50, BB_51, BB_52, BB_53, BB_54, BB_55, BB_56, BB_57, BB_58,   _, _, _, _, _, _, _, 
    BB_60, BB_61, BB_62, BB_63, BB_64, BB_65, BB_66, BB_67, BB_68,   _, _, _, _, _, _, _, 
    BB_70, BB_71, BB_72, BB_73, BB_74, BB_75, BB_76, BB_77, BB_78,   _, _, _, _, _, _, _, 
    BB_80, BB_81, BB_82, BB_83, BB_84, BB_85, BB_86, BB_87, BB_88,   _, _, _, _, _, _, _, 
    BB_90, BB_91, BB_92, BB_93, BB_94, BB_95, BB_96, BB_97, BB_98,   _, _, _, _, _, _, _, 
] = [1 << sq for sq in SQUARES]

BB_CORNERS = BB_00 | BB_08 | BB_98 | BB_90
# BB_CENTER = BB_D4 | BB_E4 | BB_D5 | BB_E5

# BB_LIGHT_SQUARES = 0x55aa_55aa_55aa_55aa
# BB_DARK_SQUARES = 0xaa55_aa55_aa55_aa55

BB_FILES = [
    BB_FILE_0,
    BB_FILE_1,
    BB_FILE_2,
    BB_FILE_3,
    BB_FILE_4,
    BB_FILE_5,
    BB_FILE_6,
    BB_FILE_7,
    BB_FILE_8,
] = [0x0001_0001_0001_0001_0001_0001_0001_0001_0001_0001 << i for i in range(9)]

BB_RANKS = [
    BB_RANK_0,
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8,
    BB_RANK_9,
] = [0x01ff << (16 * i) for i in range(10)]

# BB_BACKRANKS = BB_RANK_1 | BB_RANK_8


def lsb(bb: Bitboard) -> int:
    return (bb & -bb).bit_length() - 1

def scan_forward(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r

def msb(bb: Bitboard) -> int:
    return bb.bit_length() - 1

def scan_reversed(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb.bit_length() - 1
        yield r
        bb ^= BB_SQUARES[r]

def render(bb):
    print('  ',end='')
    for i in range(9):
        print(f'{i}  ',end='')
    print()
    for r in range(10):
        print(f'{r} ',end='')
        for c in range(9):
            if bb & 1:
                print('x  ',end='')
            else:
                print('.  ',end='')
            bb >>=1
        bb >>=7
        print()


# Python 3.10 or fallback.
popcount: Callable[[Bitboard], int] = getattr(int, "bit_count", lambda bb: bin(bb).count("1"))

def flip_vertical(bb: Bitboard) -> Bitboard:
    # Calculate the new bitboard for a 10x16 board
        # Perform bit shifts to flip vertically for a 10x16 board
    res = 0
    mask = 0xFFFF
    for i in range(10):
        res <<= 16
        res |= bb & mask
        bb >>= 16
    return res

def flip_horizontal(bb: Bitboard) -> Bitboard:
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#MirrorHorizontally
    bb = ((bb >> 1) & 0x5555_5555_5555_5555) | ((bb & 0x5555_5555_5555_5555) << 1)
    bb = ((bb >> 2) & 0x3333_3333_3333_3333) | ((bb & 0x3333_3333_3333_3333) << 2)
    bb = ((bb >> 4) & 0x0f0f_0f0f_0f0f_0f0f) | ((bb & 0x0f0f_0f0f_0f0f_0f0f) << 4)
    return bb

def flip_diagonal(bb: Bitboard) -> Bitboard:
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipabouttheDiagonal
    t = (bb ^ (bb << 28)) & 0x0f0f_0f0f_0000_0000
    bb = bb ^ t ^ (t >> 28)
    t = (bb ^ (bb << 14)) & 0x3333_0000_3333_0000
    bb = bb ^ t ^ (t >> 14)
    t = (bb ^ (bb << 7)) & 0x5500_5500_5500_5500
    bb = bb ^ t ^ (t >> 7)
    return bb

def flip_anti_diagonal(bb: Bitboard) -> Bitboard:
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipabouttheAntidiagonal
    t = bb ^ (bb << 36)
    bb = bb ^ ((t ^ (bb >> 36)) & 0xf0f0_f0f0_0f0f_0f0f)
    t = (bb ^ (bb << 18)) & 0xcccc_0000_cccc_0000
    bb = bb ^ t ^ (t >> 18)
    t = (bb ^ (bb << 9)) & 0xaa00_aa00_aa00_aa00
    bb = bb ^ t ^ (t >> 9)
    return bb


def shift_down(b: Bitboard) -> Bitboard:
    return b >> 16

def shift_2_down(b: Bitboard) -> Bitboard:
    return b >> 32

def shift_up(b: Bitboard) -> Bitboard:
    return (b << 16) & BB_ALL

def shift_2_up(b: Bitboard) -> Bitboard:
    return (b << 32) & BB_ALL

def shift_right(b: Bitboard) -> Bitboard:
    return (b << 1) & ~BB_FILE_0 & BB_ALL

def shift_2_right(b: Bitboard) -> Bitboard:
    return (b << 2) & ~BB_FILE_0 & ~BB_FILE_1 & BB_ALL
1
def shift_left(b: Bitboard) -> Bitboard:
    return (b >> 1) & ~BB_FILE_8

def shift_2_left(b: Bitboard) -> Bitboard:
    return (b >> 2) & ~BB_FILE_7 & ~BB_FILE_8

def shift_up_left(b: Bitboard) -> Bitboard:
    return (b << 15) & ~BB_FILE_8 & BB_ALL

def shift_up_right(b: Bitboard) -> Bitboard:
    return (b << 17) & ~BB_FILE_0 & BB_ALL

def shift_down_left(b: Bitboard) -> Bitboard:
    return (b >> 17) & ~BB_FILE_8

def shift_down_right(b: Bitboard) -> Bitboard:
    return (b >> 15) & ~BB_FILE_0

def shift_delta(b: Bitboard, d: int) -> Bitboard:
    if d>0:
        return b << d & BB_ALL
    return b >> -d
        
def _sliding_attacks(square: Square, occupied: Bitboard, deltas: Iterable[int], mask: Bitboard = BB_ALL) -> Bitboard:
    attacks = BB_EMPTY

    for delta in deltas:
        sq = square

        while True:
            sq += delta
            if not (0 <= sq < 160) or not (mask & BB_SQUARES[sq]): #or square_distance(sq, sq - delta) > 4
                break

            attacks |= BB_SQUARES[sq]

            if occupied & BB_SQUARES[sq]:
                break

    return attacks 

[UP,DOWN,LEFT,RIGHT] = [16,-16,-1,1]

def _step_attacks(square: Square, deltas: Iterable[int]) -> Bitboard:
    return _sliding_attacks(square, BB_ALL, deltas)

def to_squares(deltas):
    return [d[0]*16+d[1] for d in deltas]

deltas =  to_squares([(-1,-2),(-1,2),(1,-2),(1,2),(-2,-1),(-2,1),(2,-1),(2,1)])
BB_KNIGHT_ATTACKS = [_step_attacks(sq, deltas) for sq in SQUARES]
BB_HORSE_ATTACKS = BB_KNIGHT_ATTACKS

deltas =  to_squares([(-1,-2),(-1,2),(1,-2),(1,2),(-2,-1),(-2,1),(2,-1),(2,1),
                      (-1,-3),(-1,3),(1,-3),(1,3),(-3,-1),(-3,1),(3,-1),(3,1)])
BB_FALG_ATTACKS = [_step_attacks(sq, deltas) for sq in SQUARES]

BB_KING_ATTACKS = [_step_attacks(sq, [-1,1,-16,16,-15,15,-17,17]) for sq in SQUARES]

BB_GENERAL_PALACE = BB_03|BB_04|BB_05|BB_13|BB_14|BB_15|BB_23|BB_24|BB_25
BB_GENERAL_ATTACKS = [0]*(10*16)
for sq in [C03,C04,C05,C13,C14,C15,C23,C24,C25]:
    BB_GENERAL_ATTACKS[sq] = BB_KING_ATTACKS[sq] & BB_GENERAL_PALACE
                     
BB_ADVISOR_ATTACKS = BB_GENERAL_ATTACKS

# BB_PAWN_ATTACKS = [[_step_attacks(sq, deltas) for sq in SQUARES] for deltas in [[15,17], [-15,-17]]]



def _edges(square: Square) -> Bitboard:
    return (((BB_RANK_0 | BB_RANK_9) & ~BB_RANKS[square_rank(square)]) |
            ((BB_FILE_0 | BB_FILE_8) & ~BB_FILES[square_file(square)]))

def _carry_rippler(mask: Bitboard) -> Iterator[Bitboard]:
    # Carry-Rippler trick to iterate subsets of mask.
    subset = BB_EMPTY
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break

def _attack_table(deltas: List[int]) -> Tuple[List[Bitboard], List[Dict[Bitboard, Bitboard]]]:
    mask_table = []
    attack_table = []

    for square in SQUARES:
        if not (0 <= square < 160) or not (BB_ALL & BB_SQUARES[square]): #or square_distance(sq, sq - delta) > 4
            attack_table.append({0:0})
            mask_table.append(BB_EMPTY)
            continue
        attacks = {}

        # mask = _sliding_attacks(square, 0, deltas) & ~_edges(square)
        mask = _sliding_attacks(square, 0, deltas)
        for subset in _carry_rippler(mask):
            attacks[subset] = _sliding_attacks(square, subset, deltas)

        attack_table.append(attacks)
        mask_table.append(mask)

    return mask_table, attack_table

BB_DIAG_MASKS, BB_DIAG_ATTACKS = _attack_table([-15, -17, 17, 15])
BB_FILE_MASKS, BB_FILE_ATTACKS = _attack_table([-16, 16])
BB_RANK_MASKS, BB_RANK_ATTACKS = _attack_table([-1, 1])

def _rays() -> List[List[Bitboard]]:
    rays = []
    for a, bb_a in enumerate(BB_SQUARES):
        rays_row = []
        for b, bb_b in enumerate(BB_SQUARES):
            if not (bb_a & BB_ALL) or not (bb_b & BB_ALL):
                rays_row.append(BB_EMPTY)
            elif BB_DIAG_ATTACKS[a][0] & bb_b:
                rays_row.append((BB_DIAG_ATTACKS[a][0] & BB_DIAG_ATTACKS[b][0]) | bb_a | bb_b)
            elif BB_RANK_ATTACKS[a][0] & bb_b:
                rays_row.append(BB_RANK_ATTACKS[a][0] | bb_a)
            elif BB_FILE_ATTACKS[a][0] & bb_b:
                rays_row.append(BB_FILE_ATTACKS[a][0] | bb_a)
            else:
                rays_row.append(BB_EMPTY)
        rays.append(rays_row)
    return rays

BB_RAYS = _rays()
def ray(a: Square, b: Square) -> Bitboard:
    return BB_RAYS[a][b]

def between(a: Square, b: Square) -> Bitboard:
    bb = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b))
    return bb & (bb - 1)

BB_PAWN_ATTACKERS = [_step_attacks(sq, [15,17]) for sq in SQUARES]

BB_PAWN_MASKS = []
BB_PAWN_ATTACKS = []
for sq in SQUARES:
    if not (BB_ALL & BB_SQUARES[sq]):
        BB_PAWN_MASKS.append(BB_EMPTY)
        BB_PAWN_ATTACKS.append({BB_EMPTY:BB_EMPTY})
        continue
    attacks = {}
    bb = BB_SQUARES[sq]
    attack = shift_down_left(bb) | shift_down_right(bb)

    if C80 <= sq <= C88:
        forward = shift_down(bb) | shift_2_down(bb)
    else:
        forward = shift_down(bb)

    mask = attack|forward
    for attack_subset in _carry_rippler(attack):
        for forward_subset in _carry_rippler(forward):
            subset = attack_subset | forward_subset
            attacks[subset] = attack_subset| (_sliding_attacks(sq,subset,[DOWN],mask) & ~forward_subset)
    BB_PAWN_MASKS.append(mask)
    BB_PAWN_ATTACKS.append(attacks)

BB_SOLDIER_ATTACKS = []
for sq in SQUARES:
    if not (BB_ALL & BB_SQUARES[sq]):
        BB_SOLDIER_ATTACKS.append(0)
        continue

    bb = BB_SQUARES[sq]
    forward = shift_up(bb)

    side = BB_EMPTY
    if C50 <= sq:
        side = shift_left(bb) | shift_right(bb)

    mask = side|forward
    BB_SOLDIER_ATTACKS.append(mask)


BB_SOLDIER_ATTACKERS = []

for sq in SQUARES:
    if not (BB_ALL & BB_SQUARES[sq]):
        BB_SOLDIER_ATTACKERS.append(0)
        continue

    bb = BB_SQUARES[sq]
    forward = shift_down(bb)

    side = BB_EMPTY
    if sq >= C50:
        side = shift_left(bb) | shift_right(bb)

    mask = side|forward
    BB_SOLDIER_ATTACKERS.append(mask)

BB_ELEPHANT_MASKS = []
BB_ELEPHANT_ATTACKS = []
BB_ELEPHANT_LIMITS = BB_SQUARES[C02]|BB_SQUARES[C06]|BB_SQUARES[C20]|BB_SQUARES[C24]|BB_SQUARES[C28]|BB_SQUARES[C42] | BB_SQUARES[C46]
BB_ELEPHANT_LIMITS2 = 0
for bb in BB_SQUARES[:C49]:
    BB_ELEPHANT_LIMITS2 |= bb
BB_ELEPHANT_LIMITS2 &= BB_ALL
for sq in SQUARES:
    if not (BB_SQUARES[sq] & BB_ELEPHANT_LIMITS):
        BB_ELEPHANT_MASKS.append(BB_EMPTY)
        BB_ELEPHANT_ATTACKS.append({BB_EMPTY:BB_EMPTY})
        continue
    attacks = {}
    bb = BB_SQUARES[sq]
    mask = shift_down_left(bb) | shift_down_right(bb) | shift_up_left(bb) | shift_up_right(bb)
    mask &= BB_ELEPHANT_LIMITS2
    for subset in _carry_rippler(mask):
        attack = 0
        if not (subset & shift_down_left(bb)):
            attack |= shift_delta(bb,2*(DOWN+LEFT))
        if not (subset & shift_down_right(bb)):
            attack |= shift_delta(bb,2*(DOWN+RIGHT))
        if not (subset & shift_up_left(bb)):
            attack |= shift_delta(bb,2*(UP+LEFT))
        if not (subset & shift_up_right(bb)):
            attack |= shift_delta(bb,2*(UP+RIGHT))
        attack &= BB_ELEPHANT_LIMITS
        attacks[subset] = attack
    BB_ELEPHANT_MASKS.append(mask)
    BB_ELEPHANT_ATTACKS.append(attacks)

BB_CANNON_FILE_ATTACKS = []
for sq in SQUARES:
    attacks = BB_FILE_ATTACKS[sq]
    _attacks = {}
    for mask in attacks:
        attack = attacks[mask]
        temp_mask = mask & ~attack
        temp_attack = attacks[temp_mask]
        new_attack = temp_attack & temp_mask
        _attacks[mask] = (attack | new_attack) & ~(mask & attack)
    BB_CANNON_FILE_ATTACKS.append(_attacks)

BB_CANNON_RANK_ATTACKS = []
for sq in SQUARES:
    attacks = BB_RANK_ATTACKS[sq]
    _attacks = {}
    for mask in attacks:
        attack = attacks[mask]
        temp_mask = mask & ~attack
        temp_attack = attacks[temp_mask]
        new_attack = temp_attack & temp_mask
        _attacks[mask] = (attack | new_attack) & ~(mask & attack)
    BB_CANNON_RANK_ATTACKS.append(_attacks)

BB_CANNON_FILE_ATTACKERS = []
for sq in SQUARES:
    attacks = BB_FILE_ATTACKS[sq]
    _attacks = {}
    for mask in attacks:
        attack = attacks[mask]
        temp_mask = mask & ~attack
        temp_attack = attacks[temp_mask]
        # _attacks[mask] = ~attack &  ~ temp_mask & temp_attack
        _attacks[mask] = temp_mask & temp_attack
    BB_CANNON_FILE_ATTACKERS.append(_attacks)

BB_CANNON_RANK_ATTACKERS = []
for sq in SQUARES:
    attacks = BB_RANK_ATTACKS[sq]
    _attacks = {}
    for mask in attacks:
        attack = attacks[mask]
        temp_mask = mask & ~attack
        temp_attack = attacks[temp_mask]
        # _attacks[mask] = ~attack &  ~ temp_mask & temp_attack
        _attacks[mask] = temp_mask & temp_attack
    BB_CANNON_RANK_ATTACKERS.append(_attacks)

deltas = [16,32,-16,-32,-1,-2,1,2,-15,-30,-17,-34,15,30,17,34]
BB_QUEEN_MASKS = [_step_attacks(sq,deltas) for sq in SQUARES]


@dataclasses.dataclass
class Piece:
    """A piece with type and color."""

    piece_type: PieceType
    """The piece type."""

    color: Color
    """The piece color."""

    def symbol(self) -> str:
        """
        Gets the symbol ``P``, ``N``, ``B``, ``R``, ``Q`` or ``K`` for white
        pieces or the lower-case variants for the black pieces.
        """
        symbol = piece_symbol(self.piece_type)
        return symbol.upper() if self.color else symbol

    def unicode_symbol(self, *, invert_color: bool = False) -> str:
        """
        Gets the Unicode character for the piece.
        """
        symbol = self.symbol().swapcase() if invert_color else self.symbol()
        return UNICODE_PIECE_SYMBOLS[symbol]

    def __hash__(self) -> int:
        return self.piece_type + (-1 if self.color else 5)

    def __repr__(self) -> str:
        return f"Piece.from_symbol({self.symbol()!r})"

    def __str__(self) -> str:
        return self.symbol()

    def _repr_svg_(self) -> str:
        import chess.svg
        return chess.svg.piece(self, size=45)

    @classmethod
    def from_symbol(cls, symbol: str) -> Piece:
        """
        Creates a :class:`~chess.Piece` instance from a piece symbol.

        :raises: :exc:`ValueError` if the symbol is invalid.
        """
        return cls(PIECE_SYMBOLS.index(symbol.lower()), symbol.isupper())

@dataclasses.dataclass(unsafe_hash=True)
class Move:
    """
    Represents a move from a square to a square and possibly the promotion
    piece type.

    Drops and null moves are supported.
    """

    from_square: Square
    """The source square."""

    to_square: Square
    """The target square."""

    promotion: Optional[PieceType] = None
    """The promotion piece type or ``None``."""

    drop: Optional[PieceType] = None
    """The drop piece type or ``None``."""

    def uci(self) -> str:
        """
        Gets a UCI string for the move.

        For example, a move from a7 to a8 would be ``a7a8`` or ``a7a8q``
        (if the latter is a promotion to a queen).

        The UCI representation of a null move is ``0000``.
        """
        if self.drop:
            return piece_symbol(self.drop).upper() + "@" + SQUARE_NAMES[self.to_square]
        elif self.promotion:
            return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square] + piece_symbol(self.promotion)
        elif self:
            return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square]
        else:
            return "0000"

    def xboard(self) -> str:
        return self.uci() if self else "@@@@"

    def __bool__(self) -> bool:
        return bool(self.from_square or self.to_square or self.promotion or self.drop)

    def __repr__(self) -> str:
        return f"Move.from_uci({self.uci()!r})"

    def __str__(self) -> str:
        return self.uci()

    @classmethod
    def from_uci(cls, uci: str) -> Move:
        """
        Parses a UCI string.

        :raises: :exc:`InvalidMoveError` if the UCI string is invalid.
        """
        if uci == "0000":
            return cls.null()
        elif len(uci) == 4 and "@" == uci[1]:
            try:
                drop = PIECE_SYMBOLS.index(uci[0].lower())
                sq = SQUARE_NAMES.index(uci[2:])
            except ValueError:
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            return cls(sq, sq, drop=drop)
        elif 4 <= len(uci) <= 5:
            try:
                from_square = square(int(uci[0]),int(uci[1]))
                to_square = square(int(uci[2]),int(uci[3]))
                promotion = PIECE_SYMBOLS.index(uci[4]) if len(uci) == 5 else None
            except ValueError:
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            if from_square == to_square:
                raise InvalidMoveError(f"invalid uci (use 0000 for null moves): {uci!r}")
            return cls(from_square, to_square, promotion=promotion)
        else:
            raise InvalidMoveError(f"expected uci string to be of length 4 or 5: {uci!r}")

    @classmethod
    def null(cls) -> Move:
        """
        Gets a null move.

        A null move just passes the turn to the other side (and possibly
        forfeits en passant capturing). Null moves evaluate to ``False`` in
        boolean contexts.

        >>> import chess
        >>>
        >>> bool(chess.Move.null())
        False
        """
        return cls(0, 0)
    
IntoSquareSet = Union[SupportsInt, Iterable[Square]]

class SquareSet:
    """
    A set of squares.

    >>> import chess
    >>>
    >>> squares = chess.SquareSet([chess.A8, chess.A1])
    >>> squares
    SquareSet(0x0100_0000_0000_0001)

    >>> squares = chess.SquareSet(chess.BB_A8 | chess.BB_RANK_1)
    >>> squares
    SquareSet(0x0100_0000_0000_00ff)

    >>> print(squares)
    1 . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    1 1 1 1 1 1 1 1

    >>> len(squares)
    9

    >>> bool(squares)
    True

    >>> chess.B1 in squares
    True

    >>> for square in squares:
    ...     # 0 -- chess.A1
    ...     # 1 -- chess.B1
    ...     # 2 -- chess.C1
    ...     # 3 -- chess.D1
    ...     # 4 -- chess.E1
    ...     # 5 -- chess.F1
    ...     # 6 -- chess.G1
    ...     # 7 -- chess.H1
    ...     # 56 -- chess.A8
    ...     print(square)
    ...
    0
    1
    2
    3
    4
    5
    6
    7
    56

    >>> list(squares)
    [0, 1, 2, 3, 4, 5, 6, 7, 56]

    Square sets are internally represented by 64-bit integer masks of the
    included squares. Bitwise operations can be used to compute unions,
    intersections and shifts.

    >>> int(squares)
    72057594037928191

    Also supports common set operations like
    :func:`~chess.SquareSet.issubset()`, :func:`~chess.SquareSet.issuperset()`,
    :func:`~chess.SquareSet.union()`, :func:`~chess.SquareSet.intersection()`,
    :func:`~chess.SquareSet.difference()`,
    :func:`~chess.SquareSet.symmetric_difference()` and
    :func:`~chess.SquareSet.copy()` as well as
    :func:`~chess.SquareSet.update()`,
    :func:`~chess.SquareSet.intersection_update()`,
    :func:`~chess.SquareSet.difference_update()`,
    :func:`~chess.SquareSet.symmetric_difference_update()` and
    :func:`~chess.SquareSet.clear()`.
    """

    def __init__(self, squares: IntoSquareSet = BB_EMPTY) -> None:
        try:
            self.mask = squares.__int__() & BB_ALL  # type: ignore
            return
        except AttributeError:
            self.mask = 0

        # Try squares as an iterable. Not under except clause for nicer
        # backtraces.
        for square in squares:  # type: ignore
            self.add(square)

    # Set

    def __contains__(self, square: Square) -> bool:
        return bool(BB_SQUARES[square] & self.mask)

    def __iter__(self) -> Iterator[Square]:
        return scan_forward(self.mask)

    def __reversed__(self) -> Iterator[Square]:
        return scan_reversed(self.mask)

    def __len__(self) -> int:
        return popcount(self.mask)

    # MutableSet

    def add(self, square: Square) -> None:
        """Adds a square to the set."""
        self.mask |= BB_SQUARES[square]

    def discard(self, square: Square) -> None:
        """Discards a square from the set."""
        self.mask &= ~BB_SQUARES[square]

    # frozenset

    def isdisjoint(self, other: IntoSquareSet) -> bool:
        """Tests if the square sets are disjoint."""
        return not bool(self & other)

    def issubset(self, other: IntoSquareSet) -> bool:
        """Tests if this square set is a subset of another."""
        return not bool(self & ~SquareSet(other))

    def issuperset(self, other: IntoSquareSet) -> bool:
        """Tests if this square set is a superset of another."""
        return not bool(~self & other)

    def union(self, other: IntoSquareSet) -> SquareSet:
        return self | other

    def __or__(self, other: IntoSquareSet) -> SquareSet:
        r = SquareSet(other)
        r.mask |= self.mask
        return r

    def intersection(self, other: IntoSquareSet) -> SquareSet:
        return self & other

    def __and__(self, other: IntoSquareSet) -> SquareSet:
        r = SquareSet(other)
        r.mask &= self.mask
        return r

    def difference(self, other: IntoSquareSet) -> SquareSet:
        return self - other

    def __sub__(self, other: IntoSquareSet) -> SquareSet:
        r = SquareSet(other)
        r.mask = self.mask & ~r.mask
        return r

    def symmetric_difference(self, other: IntoSquareSet) -> SquareSet:
        return self ^ other

    def __xor__(self, other: IntoSquareSet) -> SquareSet:
        r = SquareSet(other)
        r.mask ^= self.mask
        return r

    def copy(self) -> SquareSet:
        return SquareSet(self.mask)

    # set

    def update(self, *others: IntoSquareSet) -> None:
        for other in others:
            self |= other

    def __ior__(self, other: IntoSquareSet) -> SquareSet:
        self.mask |= SquareSet(other).mask
        return self

    def intersection_update(self, *others: IntoSquareSet) -> None:
        for other in others:
            self &= other

    def __iand__(self, other: IntoSquareSet) -> SquareSet:
        self.mask &= SquareSet(other).mask
        return self

    def difference_update(self, other: IntoSquareSet) -> None:
        self -= other

    def __isub__(self, other: IntoSquareSet) -> SquareSet:
        self.mask &= ~SquareSet(other).mask
        return self

    def symmetric_difference_update(self, other: IntoSquareSet) -> None:
        self ^= other

    def __ixor__(self, other: IntoSquareSet) -> SquareSet:
        self.mask ^= SquareSet(other).mask
        return self

    def remove(self, square: Square) -> None:
        """
        Removes a square from the set.

        :raises: :exc:`KeyError` if the given *square* was not in the set.
        """
        mask = BB_SQUARES[square]
        if self.mask & mask:
            self.mask ^= mask
        else:
            raise KeyError(square)

    def pop(self) -> Square:
        """
        Removes and returns a square from the set.

        :raises: :exc:`KeyError` if the set is empty.
        """
        if not self.mask:
            raise KeyError("pop from empty SquareSet")

        square = lsb(self.mask)
        self.mask &= (self.mask - 1)
        return square

    def clear(self) -> None:
        """Removes all elements from this set."""
        self.mask = BB_EMPTY

    # SquareSet

    def carry_rippler(self) -> Iterator[Bitboard]:
        """Iterator over the subsets of this set."""
        return _carry_rippler(self.mask)

    def mirror(self) -> SquareSet:
        """Returns a vertically mirrored copy of this square set."""
        return SquareSet(flip_vertical(self.mask))

    def tolist(self) -> List[bool]:
        """Converts the set to a list of 64 bools."""
        result = [False] * 64
        for square in self:
            result[square] = True
        return result

    def __bool__(self) -> bool:
        return bool(self.mask)

    def __eq__(self, other: object) -> bool:
        try:
            return self.mask == SquareSet(other).mask  # type: ignore
        except (TypeError, ValueError):
            return NotImplemented

    def __lshift__(self, shift: int) -> SquareSet:
        return SquareSet((self.mask << shift) & BB_ALL)

    def __rshift__(self, shift: int) -> SquareSet:
        return SquareSet(self.mask >> shift)

    def __ilshift__(self, shift: int) -> SquareSet:
        self.mask = (self.mask << shift) & BB_ALL
        return self

    def __irshift__(self, shift: int) -> SquareSet:
        self.mask >>= shift
        return self

    def __invert__(self) -> SquareSet:
        return SquareSet(~self.mask & BB_ALL)

    def __int__(self) -> int:
        return self.mask

    def __index__(self) -> int:
        return self.mask

    def __repr__(self) -> str:
        return f"SquareSet({self.mask:#021_x})"

    def __str__(self) -> str:
        builder = []

        for square in SQUARES_180:
            mask = BB_SQUARES[square]
            builder.append("1" if self.mask & mask else ".")

            if not mask & BB_FILE_H:
                builder.append(" ")
            elif square != H1:
                builder.append("\n")

        return "".join(builder)

    def _repr_svg_(self) -> str:
        import chess.svg
        return chess.svg.board(squares=self, size=390)

    @classmethod
    def ray(cls, a: Square, b: Square) -> SquareSet:
        """
        All squares on the rank, file or diagonal with the two squares, if they
        are aligned.

        >>> import chess
        >>>
        >>> print(chess.SquareSet.ray(chess.E2, chess.B5))
        . . . . . . . .
        . . . . . . . .
        1 . . . . . . .
        . 1 . . . . . .
        . . 1 . . . . .
        . . . 1 . . . .
        . . . . 1 . . .
        . . . . . 1 . .
        """
        return cls(ray(a, b))

    @classmethod
    def between(cls, a: Square, b: Square) -> SquareSet:
        """
        All squares on the rank, file or diagonal between the two squares
        (bounds not included), if they are aligned.

        >>> import chess
        >>>
        >>> print(chess.SquareSet.between(chess.E2, chess.B5))
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . 1 . . . . .
        . . . 1 . . . .
        . . . . . . . .
        . . . . . . . .
        """
        return cls(between(a, b))

    @classmethod
    def from_square(cls, square: Square) -> SquareSet:
        """
        Creates a :class:`~chess.SquareSet` from a single square.

        >>> import chess
        >>>
        >>> chess.SquareSet.from_square(chess.A1) == chess.BB_A1
        True
        """
        return cls(BB_SQUARES[square])

BaseBoardT = TypeVar("BaseBoardT", bound="BaseBoard")

class BaseBoard:
    """
    A board representing the position of chess pieces. See
    :class:`~chess.Board` for a full board with move generation.

    The board is initialized with the standard chess starting position, unless
    otherwise specified in the optional *board_fen* argument. If *board_fen*
    is ``None``, an empty board is created.
    """

    def __init__(self, board_fen: Optional[str] = STARTING_BOARD_FEN) -> None:
        self.occupied_co = [BB_EMPTY, BB_EMPTY]

        self._reset_board()
        # if board_fen is None:
        #     self._clear_board()
        # elif board_fen == STARTING_BOARD_FEN:
        #     self._reset_board()
        # else:
        #     self._set_board_fen(board_fen)

    def _reset_board(self) -> None:
        self.pawns = BB_RANK_8
        self.knights = BB_91 | BB_97
        self.bishops = BB_92 | BB_95
        self.rooks = BB_90 | BB_98
        self.queens = BB_93 | BB_96
        self.king = BB_94

        self.soldiers = BB_30 | BB_32 | BB_34 | BB_36 | BB_38
        self.chariots = BB_00 | BB_08
        self.horses = BB_01 | BB_07
        self.elephants = BB_02 | BB_06
        self.advisors = BB_03 | BB_05
        self.general = BB_04
        self.cannons = BB_21 | BB_27
        self.flags = BB_23 | BB_25

        self.promoted = BB_EMPTY

        self.occupied_co[RED] = BB_RANK_8 | BB_RANK_9
        self.occupied_co[BLACK] = self.soldiers | self.chariots | self.horses|\
                            self.elephants | self.advisors| self.general | self.cannons | self.flags 
        self.occupied = self.occupied_co[RED] | self.occupied_co[BLACK]

    def reset_board(self) -> None:
        self._reset_board()

    def _clear_board(self) -> None:
        self.pawns = BB_EMPTY
        self.knights = BB_EMPTY
        self.bishops = BB_EMPTY
        self.rooks = BB_EMPTY
        self.queens = BB_EMPTY
        self.king = BB_EMPTY

        self.soldiers = BB_EMPTY
        self.chariots = BB_EMPTY
        self.horses = BB_EMPTY
        self.elephants = BB_EMPTY
        self.advisors = BB_EMPTY
        self.general = BB_EMPTY
        self.cannons = BB_EMPTY
        self.flags = BB_EMPTY

        self.promoted = BB_EMPTY

        self.occupied_co[RED] = BB_EMPTY
        self.occupied_co[BLACK] = BB_EMPTY
        self.occupied = BB_EMPTY

    def clear_board(self) -> None:
        """
        Clears the board.

        :class:`~chess.Board` also clears the move stack.
        """
        self._clear_board()

    def pieces_mask(self, piece_type: PieceType, color: Color) -> Bitboard:
        if piece_type == PAWN:
            bb = self.pawns
        elif piece_type == KNIGHT:
            bb = self.knights
        elif piece_type == BISHOP:
            bb = self.bishops
        elif piece_type == ROOK:
            bb = self.rooks
        elif piece_type == QUEEN:
            bb = self.queens
        elif piece_type == KING:
            bb = self.king
        elif piece_type == SOLDIER:
            bb = self.soldiers
        elif piece_type == HORSE:
            bb = self.horses
        elif piece_type == ELEPHANT:
            bb = self.elephants
        elif piece_type == CHARIOT:
            bb = self.chariots
        elif piece_type == ADVISOR:
            bb = self.advisors
        elif piece_type == GENERAL:
            bb = self.general
        elif piece_type == CANNON:
            bb = self.cannons
        elif piece_type == FLAG:
            bb = self.flags
        else:
            assert False, f"expected PieceType, got {piece_type!r}"
        return bb & self.occupied_co[color]
    
    def pieces(self, piece_type: PieceType, color: Color) -> SquareSet:
        """
        Gets pieces of the given type and color.

        Returns a :class:`set of squares <chess.SquareSet>`.
        """
        return SquareSet(self.pieces_mask(piece_type, color))
    
    def piece_at(self, square: Square) -> Optional[Piece]:
        """Gets the :class:`piece <chess.Piece>` at the given square."""
        piece_type = self.piece_type_at(square)
        if piece_type:
            mask = BB_SQUARES[square]
            color = bool(self.occupied_co[RED] & mask)
            return Piece(piece_type, color)
        else:
            return None
    
    def piece_type_at(self, square: Square) -> Optional[PieceType]:
        """Gets the piece type at the given square."""
        mask = BB_SQUARES[square]


        if not self.occupied & mask:
            return None  # Early return
        elif self.pawns & mask:
            return PAWN
        elif self.knights & mask:
            return KNIGHT
        elif self.bishops & mask:
            return BISHOP
        elif self.rooks & mask:
            return ROOK
        elif self.queens & mask:
            return QUEEN
        elif self.king & mask:
            return KING
        elif self.soldiers & mask:
            return SOLDIER
        elif self.horses & mask:
            return HORSE
        elif self.elephants & mask:
            return ELEPHANT
        elif self.chariots & mask:
            return CHARIOT
        elif self.advisors & mask:
            return ADVISOR
        elif self.general & mask:
            return GENERAL
        elif self.cannons & mask:
            return CANNON
        else:
            return FLAG

    def color_at(self, square: Square) -> Optional[Color]:  
        """Gets the color of the piece at the given square."""
        mask = BB_SQUARES[square]
        if self.occupied_co[RED] & mask:
            return RED
        elif self.occupied_co[BLACK] & mask:
            return BLACK
        else:
            return None
        
    def lord(self, color: Color) -> Optional[Square]:
        """
        Finds the king square of the given side. Returns ``None`` if there
        is no king of that color.

        In variants with king promotions, only non-promoted kings are
        considered.
        """
        if color == RED:
            bb_king = self.king
        else:
            bb_king = self.general
        return msb(bb_king) if bb_king else None
    
    def attacks_mask(self, square: Square) -> Bitboard:
        bb_square = BB_SQUARES[square]

        if bb_square & self.pawns:
            return BB_PAWN_ATTACKS[square][self.occupied & BB_PAWN_MASKS[square]]
        elif bb_square & self.knights or bb_square & self.horses:
            return BB_KNIGHT_ATTACKS[square]
        elif bb_square & self.king:
            return BB_KING_ATTACKS[square]
        elif bb_square & self.soldiers:
            return BB_SOLDIER_ATTACKS[square]
        elif bb_square & self.general:
            return BB_GENERAL_ATTACKS[square]
        elif bb_square & self.advisors:
            return BB_ADVISOR_ATTACKS[square]
        elif bb_square & self.elephants:
            return BB_ELEPHANT_ATTACKS[square][self.occupied & BB_ELEPHANT_MASKS[square]]
        elif bb_square & self.flags:
            return BB_FALG_ATTACKS[square]
        elif bb_square & self.cannons:
            return BB_CANNON_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied] |\
                   BB_CANNON_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
        elif bb_square & self.rooks or bb_square & self.chariots:
            return (BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied] |
                            BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied])
        elif bb_square & self.bishops:
            return BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
        else:
            return (BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied] | 
                    BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied] |
                    BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]) & BB_QUEEN_MASKS[square]

    def attacks(self, square: Square) -> SquareSet:
        """
        Gets the set of attacked squares from the given square.

        There will be no attacks if the square is empty. Pinned pieces are
        still attacking other squares.

        Returns a :class:`set of squares <chess.SquareSet>`.
        """
        return SquareSet(self.attacks_mask(square))
    
    def _attackers_mask(self, color: Color, square: Square, occupied: Bitboard) -> Bitboard:
        rank_pieces = BB_RANK_MASKS[square] & occupied
        file_pieces = BB_FILE_MASKS[square] & occupied
        diag_pieces = BB_DIAG_MASKS[square] & occupied

        if color == RED:
            attackers = (
            (BB_KING_ATTACKS[square] & self.king) |
            (BB_KNIGHT_ATTACKS[square] & self.knights) |
            (BB_RANK_ATTACKS[square][rank_pieces] & self.rooks) |
            (BB_FILE_ATTACKS[square][file_pieces] & self.rooks) |
            (BB_DIAG_ATTACKS[square][diag_pieces] & self.bishops) |
            (BB_PAWN_ATTACKERS[square] & self.pawns)|
            ((BB_DIAG_ATTACKS[square][diag_pieces] | BB_FILE_ATTACKS[square][file_pieces] | BB_RANK_ATTACKS[square][rank_pieces]) & BB_QUEEN_MASKS[square] & self.queens))
        else:
            attackers = (
                (BB_GENERAL_ATTACKS[square] & self.general) |
                (BB_ADVISOR_ATTACKS[square] & self.advisors) |
                (BB_HORSE_ATTACKS[square] & self.horses) |
                (BB_RANK_ATTACKS[square][rank_pieces] & self.chariots) |
                (BB_FILE_ATTACKS[square][file_pieces] & self.chariots) |
                (BB_ELEPHANT_ATTACKS[square][BB_ELEPHANT_MASKS[square] & occupied] & self.elephants) |
                (BB_SOLDIER_ATTACKERS[square] & self.soldiers) |
                (BB_FALG_ATTACKS[square] & self.flags)|
                (BB_CANNON_FILE_ATTACKERS[square][file_pieces] & self.cannons)|
                (BB_CANNON_RANK_ATTACKERS[square][rank_pieces] & self.cannons))

        return attackers & self.occupied_co[color]
    
    def attackers_mask(self, color: Color, square: Square, occupied: Optional[Bitboard] = None) -> Bitboard:
        if occupied:
            return self._attackers_mask(color, square, occupied)
        return self._attackers_mask(color, square, self.occupied)
    
    def is_attacked_by(self, color: Color, square: Square, occupied: Optional[Bitboard] = None) -> bool:
        """
        Checks if the given side attacks the given square.

        Pinned pieces still count as attackers. Pawns that can be captured
        en passant are **not** considered attacked.
        """
        return bool(self.attackers_mask(color, square, occupied))

    def attackers(self, color: Color, square: Square) -> SquareSet:
        """
        Gets the set of attackers of the given color for the given square.

        Pinned pieces still count as attackers.

        Returns a :class:`set of squares <chess.SquareSet>`.
        """
        return SquareSet(self.attackers_mask(color, square))

    def _remove_piece_at(self, square: Square) -> Optional[PieceType]:
        piece_type = self.piece_type_at(square)
        mask = BB_SQUARES[square]

        if piece_type == PAWN:
            self.pawns ^= mask
        elif piece_type == KNIGHT:
            self.knights ^= mask
        elif piece_type == BISHOP:
            self.bishops ^= mask
        elif piece_type == ROOK:
            self.rooks ^= mask
        elif piece_type == QUEEN:
            self.queens ^= mask
        elif piece_type == KING:
            self.king ^= mask
        elif piece_type == SOLDIER:
            self.soldiers ^= mask
        elif piece_type == HORSE:
            self.horses ^= mask
        elif piece_type == ELEPHANT:
            self.elephants ^= mask
        elif piece_type == CHARIOT:
            self.chariots ^= mask
        elif piece_type == ADVISOR:
            self.advisors ^= mask
        elif piece_type == GENERAL:
            self.general ^= mask
        elif piece_type == CANNON:
            self.cannons ^= mask
        elif piece_type == FLAG:
            self.flags ^= mask
        else:
            return None

        self.occupied ^= mask
        self.occupied_co[RED] &= ~mask
        self.occupied_co[BLACK] &= ~mask

        self.promoted &= ~mask

        return piece_type

    def remove_piece_at(self, square: Square) -> Optional[Piece]:
        """
        Removes the piece from the given square. Returns the
        :class:`~chess.Piece` or ``None`` if the square was already empty.

        :class:`~chess.Board` also clears the move stack.
        """
        color = bool(self.occupied_co[RED] & BB_SQUARES[square])
        piece_type = self._remove_piece_at(square)
        return Piece(piece_type, color) if piece_type else None

    def _set_piece_at(self, square: Square, piece_type: PieceType, color: Color, promoted: bool = False) -> None:
        self._remove_piece_at(square)

        mask = BB_SQUARES[square]

        if piece_type == PAWN:
            self.pawns |= mask
        elif piece_type == KNIGHT:
            self.knights |= mask
        elif piece_type == BISHOP:
            self.bishops |= mask
        elif piece_type == ROOK:
            self.rooks |= mask
        elif piece_type == QUEEN:
            self.queens |= mask
        elif piece_type == KING:
            self.king |= mask
        elif piece_type == SOLDIER:
            self.soldiers |= mask
        elif piece_type == HORSE:
            self.horses |= mask
        elif piece_type == ELEPHANT:
            self.elephants |= mask
        elif piece_type == CHARIOT:
            self.chariots |= mask
        elif piece_type == ADVISOR:
            self.advisors |= mask
        elif piece_type == GENERAL:
            self.general |= mask
        elif piece_type == CANNON:
            self.cannons |= mask
        elif piece_type == FLAG:
            self.flags |= mask
        else:
            return

        self.occupied ^= mask
        self.occupied_co[color] ^= mask

        if promoted:
            self.promoted ^= mask

    def set_piece_at(self, square: Square, piece: Optional[Piece], promoted: bool = False) -> None:
        """
        Sets a piece at the given square.

        An existing piece is replaced. Setting *piece* to ``None`` is
        equivalent to :func:`~chess.Board.remove_piece_at()`.

        :class:`~chess.Board` also clears the move stack.
        """
        if piece is None:
            self._remove_piece_at(square)
        else:
            self._set_piece_at(square, piece.piece_type, piece.color, promoted)

    def piece_map(self, *, mask: Bitboard = BB_ALL) -> Dict[Square, Piece]:
        """
        Gets a dictionary of :class:`pieces <chess.Piece>` by square index.
        """
        result = {}
        for square in scan_reversed(self.occupied & mask):
            result[square] = typing.cast(Piece, self.piece_at(square))
        return result
    
    def _set_piece_map(self, pieces: Mapping[Square, Piece]) -> None:
        self._clear_board()
        for square, piece in pieces.items():
            self._set_piece_at(square, piece.piece_type, piece.color)

    def set_piece_map(self, pieces: Mapping[Square, Piece]) -> None:
        """
        Sets up the board from a dictionary of :class:`pieces <chess.Piece>`
        by square index.

        :class:`~chess.Board` also clears the move stack.
        """
        self._set_piece_map(pieces)

    def __str__(self) -> str:
        builder = []

        for square in SQUARES_180:
            piece = self.piece_at(square)

            if piece:
                builder.append(piece.symbol())
            else:
                builder.append(".")

            if BB_SQUARES[square] & BB_FILE_8:
                if square != C08:
                    builder.append("\n")
            else:
                builder.append(" ")

        return "".join(builder)

    def unicode(self, *, invert_color: bool = False, borders: bool = False, empty_square: str = "⭘ ", orientation: Color = BLACK) -> str:
        """
        Returns a string representation of the board with Unicode pieces.
        Useful for pretty-printing to a terminal. ⭘

        :param invert_color: Invert color of the Unicode pieces.
        :param borders: Show borders and a coordinate margin.
        """
        builder = ["   ０ １ ２ ３ ４ ５ ６ ７ ８\n"]
        builder.append("  ----------------------------\n")
        for i,rank_index in enumerate(range(9, -1, -1) if orientation else range(10)):
            builder.append(f'{i} |')
            if borders:
                builder.append("  ")
                # builder.append("-" * 17)
                builder.append("\n")

                builder.append(RANK_NAMES[rank_index])
                builder.append(" ")

            for i, file_index in enumerate(range(9) if not orientation else range(8, -1, -1)):
                square_index = square(rank_index,file_index)

                if borders:
                    builder.append("|")
                elif i > 0:
                    builder.append(" ")

                piece = self.piece_at(square_index)

                if piece:
                    builder.append(piece.unicode_symbol(invert_color=invert_color))
                else:
                    builder.append(empty_square)

            builder.append('|')

            if(rank_index > 0 if orientation else rank_index < 10):
                builder.append("\n")

        builder.append('\n  ----------------------------\n')
        return "".join(builder) 
    
    def __eq__(self, board: object) -> bool:
        if isinstance(board, BaseBoard):
            return (
                self.occupied == board.occupied and
                self.occupied_co[RED] == board.occupied_co[RED] and
                self.pawns == board.pawns and
                self.knights == board.knights and
                self.bishops == board.bishops and
                self.rooks == board.rooks and
                self.queens == board.queens and
                self.king == board.king and
                self.soldiers == board.soldiers and
                self.horses == board.horses and
                self.elephants == board.elephants and
                self.chariots == board.chariots and
                self.advisors == board.advisors and
                self.general == board.general and
                self.cannons == board.cannons and
                self.flags == board.flags)
        else:
            return NotImplemented

    def apply_transform(self, f: Callable[[Bitboard], Bitboard]) -> None:
        self.pawns = f(self.pawns)
        self.knights = f(self.knights)
        self.bishops = f(self.bishops)
        self.rooks = f(self.rooks)
        self.queens = f(self.queens)
        self.king = f(self.king)
        self.soldiers = f(self.soldiers)
        self.horses = f(self.horses)
        self.elephants = f(self.elephants)
        self.chariots = f(self.chariots)
        self.advisors = f(self.advisors)
        self.general = f(self.general)
        self.cannons = f(self.cannons)
        self.flags = f(self.flags)

        self.occupied_co[RED] = f(self.occupied_co[RED])
        self.occupied_co[BLACK] = f(self.occupied_co[BLACK])
        self.occupied = f(self.occupied)
        self.promoted = f(self.promoted)

    def apply_mirror(self: BaseBoardT) -> None:
        self.apply_transform(flip_vertical)
        self.occupied_co[RED], self.occupied_co[BLACK] = self.occupied_co[BLACK], self.occupied_co[RED]

    def mirror(self: BaseBoardT) -> BaseBoardT:
        """
        Returns a mirrored copy of the board (without move stack).

        The board is mirrored vertically and piece colors are swapped, so that
        the position is equivalent modulo color.

        Alternatively, :func:`~chess.BaseBoard.apply_mirror()` can be used
        to mirror the board.
        """
        board = self.copy()
        board.apply_mirror()
        return board

    def copy(self: BaseBoardT) -> BaseBoardT:
        """Creates a copy of the board."""
        board = type(self)(None)
        
        board.pawns = self.pawns
        board.knights = self.knights
        board.bishops = self.bishops
        board.rooks = self.rooks
        board.queens = self.queens
        board.king = self.king
        board.soldiers = self.soldiers
        board.horses = self.horses
        board.elephants = self.elephants
        board.chariots = self.chariots
        board.advisors = self.advisors
        board.general = self.general
        board.cannons = self.cannons
        board.flags = self.flags

        board.occupied_co[RED] = self.occupied_co[RED]
        board.occupied_co[BLACK] = self.occupied_co[BLACK]
        board.occupied = self.occupied
        board.promoted = self.promoted

        board.turn = self.turn

        return board
    
    def __copy__(self: BaseBoardT) -> BaseBoardT:
        return self.copy()

    def __deepcopy__(self: BaseBoardT, memo: Dict[int, object]) -> BaseBoardT:
        board = self.copy()
        memo[id(self)] = board
        return board
    
    @classmethod
    def empty(cls: Type[BaseBoardT]) -> BaseBoardT:
        """
        Creates a new empty board. Also see
        :func:`~chess.BaseBoard.clear_board()`.
        """
        return cls(None)
    
BoardT = TypeVar("BoardT", bound="Board")

class _BoardState(Generic[BoardT]):

    def __init__(self, board: BoardT) -> None:
        self.pawns = board.pawns
        self.knights = board.knights
        self.bishops = board.bishops
        self.rooks = board.rooks
        self.queens = board.queens
        self.king = board.king
        self.soldiers = board.soldiers
        self.horses = board.horses
        self.elephants = board.elephants
        self.chariots = board.chariots
        self.advisors = board.advisors
        self.general = board.general
        self.cannons = board.cannons
        self.flags = board.flags

        self.occupied_w = board.occupied_co[RED]
        self.occupied_b = board.occupied_co[BLACK]
        self.occupied = board.occupied

        self.promoted = board.promoted

        self.turn = board.turn
        self.halfmove_clock = board.halfmove_clock
        self.fullmove_number = board.fullmove_number

    def restore(self, board: BoardT) -> None:
        board.pawns = self.pawns
        board.knights = self.knights
        board.bishops = self.bishops
        board.rooks = self.rooks
        board.queens = self.queens
        board.king = self.king
        board.soldiers = self.soldiers
        board.horses = self.horses
        board.elephants = self.elephants
        board.chariots = self.chariots
        board.advisors = self.advisors
        board.general = self.general
        board.cannons = self.cannons
        board.flags = self.flags

        board.occupied_co[RED] = self.occupied_w
        board.occupied_co[BLACK] = self.occupied_b
        board.occupied = self.occupied

        board.promoted = self.promoted

        board.turn = self.turn
        board.halfmove_clock = self.halfmove_clock
        board.fullmove_number = self.fullmove_number

class LegalMoveGenerator:

    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        return any(self.board.generate_legal_moves())

    def count(self) -> int:
        # List conversion is faster than iterating.
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        return self.board.generate_legal_moves()

    def __contains__(self, move: Move) -> bool:
        return self.board.is_legal(move)

    def __repr__(self) -> str:
        sans = ", ".join(self.board.san(move) for move in self)
        return f"<LegalMoveGenerator at {id(self):#x} ({sans})>"

class PseudoLegalMoveGenerator:

    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        return any(self.board.generate_pseudo_legal_moves())

    def count(self) -> int:
        # List conversion is faster than iterating.
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        return self.board.generate_pseudo_legal_moves()

    def __contains__(self, move: Move) -> bool:
        return self.board.is_pseudo_legal(move)

    def __repr__(self) -> str:
        builder = []

        for move in self:
            if self.board.is_legal(move):
                builder.append(self.board.san(move))
            else:
                builder.append(self.board.uci(move))

        sans = ", ".join(builder)
        return f"<PseudoLegalMoveGenerator at {id(self):#x} ({sans})>"

class Board(BaseBoard):
    """
    A :class:`~chess.BaseBoard`, additional information representing
    a chess position, and a :data:`move stack <chess.Board.move_stack>`.

    Provides :data:`move generation <chess.Board.legal_moves>`, validation,
    :func:`parsing <chess.Board.parse_san()>`, attack generation,
    :func:`game end detection <chess.Board.is_game_over()>`,
    and the capability to :func:`make <chess.Board.push()>` and
    :func:`unmake <chess.Board.pop()>` moves.

    The board is initialized to the standard chess starting position,
    unless otherwise specified in the optional *fen* argument.
    If *fen* is ``None``, an empty board is created.

    Optionally supports *chess960*. In Chess960, castling moves are encoded
    by a king move to the corresponding rook square.
    Use :func:`chess.Board.from_chess960_pos()` to create a board with one
    of the Chess960 starting positions.

    It's safe to set :data:`~Board.turn`, :data:`~Board.castling_rights`,
    :data:`~Board.ep_square`, :data:`~Board.halfmove_clock` and
    :data:`~Board.fullmove_number` directly.

    .. warning::
        It is possible to set up and work with invalid positions. In this
        case, :class:`~chess.Board` implements a kind of "pseudo-chess"
        (useful to gracefully handle errors or to implement chess variants).
        Use :func:`~chess.Board.is_valid()` to detect invalid positions.
    """

    turn: Color
    """The side to move (``chess.WHITE`` or ``chess.BLACK``)."""

    fullmove_number: int
    """
    Counts move pairs. Starts at `1` and is incremented after every move
    of the black side.
    """

    halfmove_clock: int
    """The number of half-moves since the last capture or pawn move."""

    promoted: Bitboard
    """A bitmask of pieces that have been promoted."""

    move_stack: List[Move]
    """
    The move stack. Use :func:`Board.push() <chess.Board.push()>`,
    :func:`Board.pop() <chess.Board.pop()>`,
    :func:`Board.peek() <chess.Board.peek()>` and
    :func:`Board.clear_stack() <chess.Board.clear_stack()>` for
    manipulation.
    """

    def __init__(self: BoardT, fen: Optional[str] = STARTING_FEN) -> None:
        BaseBoard.__init__(self, None)
        self.move_stack = []
        self._stack: List[_BoardState[BoardT]] = []

        self.reset()
        # if fen is None:
        #     self.clear()
        # elif fen == type(self).starting_fen:
        #     self.reset()
        # else:
        #     self.set_fen(fen)

    @property
    def legal_moves(self) -> LegalMoveGenerator:
        """
        A dynamic list of legal moves.

        >>> import chess
        >>>
        >>> board = chess.Board()
        >>> board.legal_moves.count()
        20
        >>> bool(board.legal_moves)
        True
        >>> move = chess.Move.from_uci("g1f3")
        >>> move in board.legal_moves
        True

        Wraps :func:`~chess.Board.generate_legal_moves()` and
        :func:`~chess.Board.is_legal()`.
        """
        return LegalMoveGenerator(self)
    
    @property
    def pseudo_legal_moves(self) -> PseudoLegalMoveGenerator:
        """
        A dynamic list of pseudo-legal moves, much like the legal move list.

        Pseudo-legal moves might leave or put the king in check, but are
        otherwise valid. Null moves are not pseudo-legal. Castling moves are
        only included if they are completely legal.

        Wraps :func:`~chess.Board.generate_pseudo_legal_moves()` and
        :func:`~chess.Board.is_pseudo_legal()`.
        """
        return PseudoLegalMoveGenerator(self)

    def reset(self) -> None:
        """Restores the starting position."""
        self.turn = BLACK
        self.halfmove_clock = 0
        self.fullmove_number = 1

        self.reset_board()

    def reset_board(self) -> None:
        super().reset_board()
        self.clear_stack()

    def clear(self) -> None:
        """
        Clears the board.

        Resets move stack and move counters. The side to move is white. There
        are no rooks or kings, so castling rights are removed.

        In order to be in a valid :func:`~chess.Board.status()`, at least kings
        need to be put on the board.
        """
        self.turn = RED
        self.halfmove_clock = 0
        self.fullmove_number = 1

        self.clear_board()

    def clear_board(self) -> None:
        super().clear_board()
        self.clear_stack()

    def clear_stack(self) -> None:
        """Clears the move stack."""
        self.move_stack.clear()
        self._stack.clear()

    def root(self: BoardT) -> BoardT:
        """Returns a copy of the root position."""
        if self._stack:
            board = type(self)(None)
            self._stack[0].restore(board)
            return board
        else:
            return self.copy(stack=False)
    
    def ply(self) -> int:
        """
        Returns the number of half-moves since the start of the game, as
        indicated by :data:`~chess.Board.fullmove_number` and
        :data:`~chess.Board.turn`.

        If moves have been pushed from the beginning, this is usually equal to
        ``len(board.move_stack)``. But note that a board can be set up with
        arbitrary starting positions, and the stack can be cleared.
        """
        return 2 * (self.fullmove_number - 1) + (self.turn == BLACK)

    def remove_piece_at(self, square: Square) -> Optional[Piece]:
        piece = super().remove_piece_at(square)
        self.clear_stack()
        return piece
    
    def set_piece_at(self, square: Square, piece: Optional[Piece], promoted: bool = False) -> None:
        super().set_piece_at(square, piece, promoted=promoted)
        self.clear_stack()

    def generate_pseudo_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        our_pieces = self.occupied_co[self.turn]

        # Generate piece moves.
        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in scan_reversed(non_pawns):
            moves = self.attacks_mask(from_square) & ~our_pieces & to_mask
            for to_square in scan_reversed(moves):
                yield Move(from_square, to_square)

        # The remaining moves are all pawn moves.
        pawns = self.pawns & self.occupied_co[self.turn] & from_mask
        if not pawns:
            return

        # Generate pawn captures.
        for from_square in scan_reversed(pawns):
            targets = (
                BB_PAWN_ATTACKS[from_square][BB_PAWN_MASKS[from_square] & self.occupied] &
                ~self.occupied_co[self.turn] & to_mask)

            for to_square in scan_reversed(targets):
                if square_rank(to_square) in [0, 9]:
                    yield Move(from_square, to_square, QUEEN)
                    # yield Move(from_square, to_square, ROOK)
                    # yield Move(from_square, to_square, BISHOP)
                    # yield Move(from_square, to_square, KNIGHT)
                else:
                    yield Move(from_square, to_square)

    def checkers_mask(self) -> Bitboard:
        king = self.lord(self.turn)
        return BB_EMPTY if king is None else self.attackers_mask(not self.turn, king)

    def checkers(self) -> SquareSet:
        """
        Gets the pieces currently giving check.

        Returns a :class:`set of squares <chess.SquareSet>`.
        """
        return SquareSet(self.checkers_mask())

    def is_check(self) -> bool:
        """Tests if the current side to move is in check."""
        return bool(self.checkers_mask())

    def is_into_check(self, move: Move) -> bool:
        king = self.lord(self.turn)
        if king is None:
            return False

        # If already in check, look if it is an evasion.
        checkers = self.attackers_mask(not self.turn, king)
        if checkers and move not in self._generate_evasions(king, checkers, BB_SQUARES[move.from_square], BB_SQUARES[move.to_square]):
            return True

        return not self._is_safe(king, self._slider_blockers(king), self._restriction(king), move)

    def _generate_evasions(self, king: Square, checkers: Bitboard, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        sliders = checkers & (self.bishops | self.rooks | self.chariots)

        attacked = 0
        for checker in scan_reversed(sliders):
            attacked |= ray(king, checker) & ~BB_SQUARES[checker]

        cannonfuse = 0
        for checker in scan_reversed(checkers & self.cannons):
            cannonfuse = (between(king,checker) & self.occupied)
            attacked |= ray(king,checker) & ~cannonfuse
        
        for checker in scan_reversed(checkers & self.queens):
            attacked |= BB_QUEEN_MASKS[checker] & BB_DIAG_ATTACKS[checker][self.occupied & BB_DIAG_MASKS[checker]]
        

        if BB_SQUARES[king] & from_mask:
            if self.turn == RED:
                BB_ATTACKS = BB_KING_ATTACKS
            else:
                BB_ATTACKS = BB_GENERAL_ATTACKS
            for to_square in scan_reversed(BB_ATTACKS[king] & ~self.occupied_co[self.turn] & ~attacked & to_mask):
                yield Move(king, to_square)

        checker = msb(checkers)
        if BB_SQUARES[checker] == checkers:
            if checkers & self.cannons:
                # Capture or block a single checker.
                target = (between(king, checker) | checkers) & ~cannonfuse
                target2 = between(king, checker)
                yield from self.generate_pseudo_legal_moves(~king & from_mask & ~cannonfuse, target & to_mask)
                yield from self.generate_pseudo_legal_moves(cannonfuse & from_mask, ~target2 & to_mask)
            else:
                # Capture or block a single checker.
                # render(cannonfuse)
                target = (between(king, checker) | checkers)
                yield from self.generate_pseudo_legal_moves(~king & from_mask, target & to_mask)              

    def _slider_blockers(self, king: Square) -> Bitboard:
        snipers = ((BB_RANK_ATTACKS[king][0] & (self.rooks | self.chariots)) |
                   (BB_FILE_ATTACKS[king][0] & (self.rooks | self.chariots)) |
                   (BB_DIAG_ATTACKS[king][0] & (self.bishops)) |
                   (BB_QUEEN_MASKS[king] & self.queens) |
                   (BB_ELEPHANT_ATTACKS[king][0] & self.elephants)) 

        blockers = 0
        for sniper in scan_reversed(snipers & self.occupied_co[not self.turn]):
            b = between(king, sniper) & self.occupied

            # Add to blockers if exactly one piece in-between.
            if b and BB_SQUARES[msb(b)] == b:
                blockers |= b

        snipers = ((BB_RANK_ATTACKS[king][0] & self.cannons) |
                   (BB_FILE_ATTACKS[king][0] & self.cannons))
        cannon_blockers = 0
        for sniper in scan_reversed(snipers & self.occupied_co[not self.turn]):
            b = between(king, sniper) & self.occupied

            if len(list(scan_reversed(b))) == 2:
                cannon_blockers |= b

        return blockers & self.occupied_co[self.turn], cannon_blockers

    def _restriction(self,king: Square) -> Bitboard:
        cannons = (BB_FILE_MASKS[king] | BB_RANK_MASKS[king]) & self.cannons
        res = 0
        for c in scan_reversed(cannons):
            b = between(king,c)
            if not b & self.occupied:
                res |= b
        return res

    def gives_check(self, move: Move) -> bool:
        """
        Probes if the given move would put the opponent in check. The move
        must be at least pseudo-legal.
        """
        self.push(move)
        try:
            return self.is_check()
        finally:
            self.pop()

    def is_pseudo_legal(self, move: Move) -> bool:
        # Null moves are not pseudo-legal.
        if not move:
            return False

        # Drops are not pseudo-legal.
        if move.drop:
            return False

        # Source square must not be vacant.
        piece = self.piece_type_at(move.from_square)
        if not piece:
            return False

        # Get square masks.
        from_mask = BB_SQUARES[move.from_square]
        to_mask = BB_SQUARES[move.to_square]

        # Check turn.
        if not self.occupied_co[self.turn] & from_mask:
            return False

        # Only pawns can promote and only on the backrank.
        if move.promotion:
            if piece != PAWN:
                return False

            if square_rank(move.to_square) != 0:
                return False

        # Destination square can not be occupied.
        if self.occupied_co[self.turn] & to_mask:
            return False

        # Handle all other pieces.
        return bool(self.attacks_mask(move.from_square) & to_mask)
    
    def is_legal(self, move: Move) -> bool:
        return not self.is_variant_end() and self.is_pseudo_legal(move) and not self.is_into_check(move)
    
    def generate_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        if self.is_variant_end():
            return
        king = self.lord(self.turn)
        if king:
            blockers = self._slider_blockers(king)
            restriction = self._restriction(king)
            checkers = self.attackers_mask(not self.turn, king)
            if checkers:
                for move in self._generate_evasions(king, checkers, from_mask, to_mask):
                    if self._is_safe(king, blockers, restriction, move):
                        yield move
            else:
                for move in self.generate_pseudo_legal_moves(from_mask, to_mask):
                    if self._is_safe(king, blockers, restriction, move):
                        yield move
        else:
            yield from self.generate_pseudo_legal_moves(from_mask, to_mask)

    def _is_safe(self, king: Square, blockers: Tuple[Bitboard], restriction: Bitboard, move: Move) -> bool:
        if move.from_square == king:
            return not self.is_attacked_by(not self.turn, move.to_square, self.occupied & ~BB_SQUARES[king])
        else:
            return bool(not BB_SQUARES[move.to_square] & restriction and
                        (   
                            not (blockers[0] | blockers[1]) & BB_SQUARES[move.from_square] or 
                            (   
                                ray(move.from_square, move.to_square) & BB_SQUARES[king] and 
                                not BB_SQUARES[move.to_square] & blockers[1]
                            )
                        )
                    )
    
    def is_variant_end(self) -> bool:
        """
        Checks if the game is over due to a special variant end condition.

        Note, for example, that stalemate is not considered a variant-specific
        end condition (this method will return ``False``), yet it can have a
        special **result** in suicide chess (any of
        :func:`~chess.Board.is_variant_loss()`,
        :func:`~chess.Board.is_variant_win()`,
        :func:`~chess.Board.is_variant_draw()` might return ``True``).
        """
        return False

    def result(self, *, claim_draw: bool = False) -> str:
        outcome = self.outcome(claim_draw=claim_draw)
        return outcome.result() if outcome else "*"
    
    def is_game_over(self, *, claim_draw: bool = False) -> bool:
        return self.outcome(claim_draw=claim_draw) is not None
    
    def outcome(self, *, claim_draw: bool = False) -> Optional[Outcome]:
        """
        Checks if the game is over due to
        :func:`checkmate <chess.Board.is_checkmate()>`,
        :func:`stalemate <chess.Board.is_stalemate()>`,
        :func:`insufficient material <chess.Board.is_insufficient_material()>`,
        the :func:`seventyfive-move rule <chess.Board.is_seventyfive_moves()>`,
        :func:`fivefold repetition <chess.Board.is_fivefold_repetition()>`,
        or a :func:`variant end condition <chess.Board.is_variant_end()>`.
        Returns the :class:`chess.Outcome` if the game has ended, otherwise
        ``None``.

        Alternatively, use :func:`~chess.Board.is_game_over()` if you are not
        interested in who won the game and why.

        The game is not considered to be over by the
        :func:`fifty-move rule <chess.Board.can_claim_fifty_moves()>` or
        :func:`threefold repetition <chess.Board.can_claim_threefold_repetition()>`,
        unless *claim_draw* is given. Note that checking the latter can be
        slow.
        """
        # # Variant support.
        # if self.is_variant_loss():
        #     return Outcome(Termination.VARIANT_LOSS, not self.turn)
        # if self.is_variant_win():
        #     return Outcome(Termination.VARIANT_WIN, self.turn)
        # if self.is_variant_draw():
        #     return Outcome(Termination.VARIANT_DRAW, None)

        # Normal game end.
        # if self.is_checkmate():
        #     return Outcome(Termination.CHECKMATE, not self.turn)
        # if self.is_insufficient_material():
        #     return Outcome(Termination.INSUFFICIENT_MATERIAL, None)
        if not any(self.generate_legal_moves()):
            return Outcome(Termination.CHECKMATE,not self.turn)

        # # Automatic draws.
        if self.is_seventyfive_moves():
            return Outcome(Termination.SEVENTYFIVE_MOVES, None)
        # if self.is_fivefold_repetition():
        #     return Outcome(Termination.FIVEFOLD_REPETITION, None)

        # # Claimable draws.
        # if claim_draw:
        #     if self.can_claim_fifty_moves():
        #         return Outcome(Termination.FIFTY_MOVES, None)
        #     if self.can_claim_threefold_repetition():
        #         return Outcome(Termination.THREEFOLD_REPETITION, None)

        return None
    
    def is_seventyfive_moves(self) -> bool:
        """
        Since the 1st of July 2014, a game is automatically drawn (without
        a claim by one of the players) if the half-move clock since a capture
        or pawn move is equal to or greater than 150. Other means to end a game
        take precedence.
        """
        return self._is_halfmoves(150)
    
    def _is_halfmoves(self, n: int) -> bool:
        return self.halfmove_clock >= n and any(self.generate_legal_moves())

    def _board_state(self: BoardT) -> _BoardState[BoardT]:
        return _BoardState(self)

    def is_zeroing(self, move: Move) -> bool:
        """Checks if the given pseudo-legal move is a capture or pawn move."""
        touched = BB_SQUARES[move.from_square] ^ BB_SQUARES[move.to_square]
        return bool(touched & self.occupied_co[not self.turn])

    def push(self: BoardT, move: Move) -> None:
        """
        Updates the position with the given *move* and puts it onto the
        move stack.

        >>> import chess
        >>>
        >>> board = chess.Board()
        >>>
        >>> Nf3 = chess.Move.from_uci("g1f3")
        >>> board.push(Nf3)  # Make the move

        >>> board.pop()  # Unmake the last move
        Move.from_uci('g1f3')

        Null moves just increment the move counters, switch turns and forfeit
        en passant capturing.

        .. warning::
            Moves are not checked for legality. It is the caller's
            responsibility to ensure that the move is at least pseudo-legal or
            a null move.
        """
        # Push move and remember board state.
        board_state = self._board_state()
        self.move_stack.append(move)
        self._stack.append(board_state)


        # Increment move counters.
        self.halfmove_clock += 1
        if self.turn == BLACK:
            self.fullmove_number += 1

        # On a null move, simply swap turns and reset the en passant square.
        if not move:
            self.turn = not self.turn
            return

        # Drops.
        if move.drop:
            self._set_piece_at(move.to_square, move.drop, self.turn)
            self.turn = not self.turn
            return

        # Zero the half-move clock.
        if self.is_zeroing(move):
            self.halfmove_clock = 0

        from_bb = BB_SQUARES[move.from_square]

        promoted = bool(self.promoted & from_bb)
        piece_type = self._remove_piece_at(move.from_square)
        assert piece_type is not None, f"push() expects move to be pseudo-legal, but got {move}"

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion


        self._set_piece_at(move.to_square, piece_type, self.turn, promoted)
        self.turn = not self.turn

    def pop(self: BoardT) -> Move:
        """
        Restores the previous position and returns the last move from the stack.

        :raises: :exc:`IndexError` if the move stack is empty.
        """
        move = self.move_stack.pop()
        self._stack.pop().restore(self)
        return move

import time
import random

if __name__ == '__main__':
    p = C46
    for mask in BB_ELEPHANT_ATTACKS[p]:
        print('--------------')
        render(mask)
        render(BB_ELEPHANT_ATTACKS[p][mask])

