import cocos
from cocos.scene import Scene
from cocos.layer import Layer
from cocos.menu import MenuItem, Menu
from pyglet.window import key

from gameplay import ChessBoard

import cchess

WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768

class MainMenu(cocos.layer.Layer):
    is_event_handler = True
    def __init__(self):
        super(MainMenu, self).__init__()

        items = [
            MenuItem('Player vs Player', self.on_select_mode, 1),
            MenuItem('Player vs AI', self.on_select_mode, 2),
            MenuItem('Exit', self.on_exit)
        ]
        menu = cocos.menu.Menu()
        menu.font_title['font_name'] = 'Arial'
        menu.font_title['font_size'] = 72
        menu.font_item['font_name'] = 'Arial'
        menu.font_item['font_size'] = 32
        menu.menu_valign = cocos.menu.BOTTOM
        menu.menu_halign = cocos.menu.CENTER
        menu.create_menu(items)
        menu.position = 0, WINDOW_HEIGHT //2
        self.add(menu)

    def on_select_mode(self, mode):
        # if mode == 1:
        #     cocos.director.director.push(cocos.scene.Scene(PlayerVsPlayerScene()))
        # elif mode == 2:
        #     cocos.director.director.push(cocos.scene.Scene(PlayerVsAIScene()))

        if mode == 1:
            cocos.director.director.push(cocos.scene.Scene(PlayerVsPlayerScene()))
        elif mode == 2:
            cocos.director.director.push(cocos.scene.Scene(Menu1()))

class Menu1(cocos.layer.Layer):
    is_event_handler = True
    def __init__(self):
        super().__init__()

        items = [
            MenuItem('Easy', self.on_select_mode, 0),
            MenuItem('Medium', self.on_select_mode, 1),
            MenuItem('Hard', self.on_select_mode,2)
        ]
        menu = cocos.menu.Menu()
        menu.font_title['font_name'] = 'Arial'
        menu.font_title['font_size'] = 72
        menu.font_item['font_name'] = 'Arial'
        menu.font_item['font_size'] = 32
        menu.menu_valign = cocos.menu.BOTTOM
        menu.menu_halign = cocos.menu.CENTER
        menu.create_menu(items)
        self.add(menu)

    def on_select_mode(self, mode): 
        cocos.director.director.push(cocos.scene.Scene(PlayerVsAIScene(mode)))

    # def on_exit(self):
    #     cocos.director.director.pop()
import time 
class PlayerVsPlayerScene(cocos.layer.Layer):
    is_event_handler = True
    def __init__(self):
        super().__init__()
        text = "Connecting..."
        self.label = cocos.text.Label(text, font_name='Times New Roman', font_size=40, anchor_x='center', anchor_y='center')
            # set the position of our text to x:320 y:240
        self.label.position = (600, 500)
            # add our label as a child. It is a CocosNode object, which know how to render themselves.
        self.add(self.label)


    def on_key_press(self, symbol, modifiers):
        if self.label:
            self.remove(self.label)
            self.label = None
            self.cb: ChessBoard = ChessBoard()
        self.cb.position = 100,100
        self.add(self.cb)
        self.player_side = cchess.BLACK
        self.agent = MCTSGameController()
        if symbol == key.ESCAPE:
            cocos.director.director.pop()

    def on_mouse_press(self, x, y, buttons, modifiers):
        if self.cb.selected_piece:
            if self.cb.can_move((x,y)):
                if self.check_end_game():
                    return
                self.AI_run()
                if self.check_end_game():
                    return
                self.cb.deselect_piece()
                return
            if self.cb.is_clicked((x,y),self.player_side):
                self.cb.select_piece((x,y))   
        else:
             if self.cb.is_clicked((x,y),self.player_side):
                self.cb.select_piece((x,y))

    def check_end_game(self):
        if self.cb.board.is_game_over():
            res  = self.cb.board.result()
            if res == '1/2-1/2':
                text = "Draw"
            elif res == '0-1':
                text = "You win"
            else:
                text = "You lose"

            label = cocos.text.Label(text, font_name='Times New Roman', font_size=40, anchor_x='center', anchor_y='center')
            # set the position of our text to x:320 y:240
            label.position = (1000, 500)
            # add our label as a child. It is a CocosNode object, which know how to render themselves.
            self.add(label)
            return True
        else:
            return False
        

    def AI_run(self):
        move = self.agent.get_next_move(self.cb.board)
        self.cb.move(move)

from agent import MCTSGameController
class PlayerVsAIScene(cocos.layer.Layer):
    is_event_handler = True
    def __init__(self,mode=0):
        super(PlayerVsAIScene, self).__init__()
        self.cb: ChessBoard = ChessBoard()
        self.cb.position = 100,100
        self.add(self.cb)
        self.player_side = cchess.BLACK
        if mode == 0:
            self.time = 5
        if mode == 1:
            self.time = 20
        if mode == 2:
            self.time = 30
        self.agent = MCTSGameController()


    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            cocos.director.director.pop()

    def on_mouse_press(self, x, y, buttons, modifiers):
        if self.cb.selected_piece:
            if self.cb.can_move((x,y)):
                if self.check_end_game():
                    return
                self.AI_run()
                if self.check_end_game():
                    return
                self.cb.deselect_piece()
                return
            if self.cb.is_clicked((x,y),self.player_side):
                self.cb.select_piece((x,y))   
        else:
             if self.cb.is_clicked((x,y),self.player_side):
                self.cb.select_piece((x,y))

    def check_end_game(self):
        if self.cb.board.is_game_over():
            res  = self.cb.board.result()
            if res == '1/2-1/2':
                text = "Draw"
            elif res == '0-1':
                text = "You win"
            else:
                text = "You lose"

            label = cocos.text.Label(text, font_name='Times New Roman', font_size=40, anchor_x='center', anchor_y='center')
            # set the position of our text to x:320 y:240
            label.position = (1000, 500)
            # add our label as a child. It is a CocosNode object, which know how to render themselves.
            self.add(label)
            return True
        else:
            return False
        

    def AI_run(self):
        move = self.agent.get_next_move(self.cb.board,self.time)
        self.cb.move(move)



# Add your ChessboardLayer and AIChessboardLayer implementations here...

if __name__ == "__main__":
    cocos.director.director.init(width =WINDOW_WIDTH,height = WINDOW_HEIGHT)
    cocos.director.director.run(cocos.scene.Scene(MainMenu()))
