from operator import xor
import pygame
import pygame_menu
import numpy as np
import random


### INITIALIZING VARIABLES

pygame.init()

### Graphics
mappic = pygame.image.load("pics\HSG_Map_Outline.jpg")
mapnewpic = pygame.image.load("pics\MapGrass_Progress_Street.png")
mapstartvector = np.array([41, 35])
sturmpic = pygame.image.load("pics\Sturm30.png")
studentmpic = pygame.image.load("pics\mann_30.png")
win = pygame.display.set_mode()
pygame.display.set_caption("Insert name of game here")
map = [
    "000000000000000000000000000000",    
    "000000000010000000000000000000",
    "000000000110000000000011000000",
    "000000000010100000001110000000",
    "000000001111111111111000000000",
    "000000001111111000000000000000",
    "000011111100011000011110000000",        
    "000000011101011010000100010000",   
    "000000010011111111000111111110",
    "000000010111111111000100000000",                         
    "000000010011111111111100000000",                    
    "000011111111111111111100000000",
    "000000001110000011011100000000",
    "000000001110000111001100000000",
    "000000001110000001001100000000",
    "000000001110000001001100000000", 
    "000000000100000001111100000000",              
    "000000000100000001000100000000",   
    "000000000100000001000100100000",
    "000000000111111111111111111110"                  
]

mapwidth = 30
mapheight = 20
run = True
score = 0
GameOver = False
IsMenuScreen = False

# Student stuff
StudentVec = np.array([12,3])
StudentChange = np.array([1,0])

# Sturm stuff
SturmVec = np.array([24,18])
SturmChange = np.array([-1,0])

# Token stuff
TokenVec = np.array([3,0])
ListOfTokenSpawnPoints = [[9,2],[12,3],[22,6],[4,6],[4,11],[11,7], [16,7], [25,7], [9,9], [15,13], [19,12], [24,18]]

def set_difficulty(value, difficulty):
    # insert code here
    pass

def start_the_game():
    global IsMenuScreen
    IsMenuScreen = False


def set_name(value, name):
    # insert code here
    pass
menu = pygame_menu.Menu(600, 900, 'Startwoche 2021 – Introduction Game!',
                       theme=pygame_menu.themes.THEME_GREEN)
    # I am not sure if THEME_GREEN exists, since I cannot get it to run on my machine. Default Theme is "BLUE"
menu.add_text_input('This game will familiarize you with the premises of the HSG,')
menu.add_text_input('without needing to visit the university in person.')
menu.add_text_input('Get all the stuff before Sturm catches you!')
    # documentation on text inserts
    # https://pygame-menu.readthedocs.io/en/3.5.6/_source/widgets_textinput.html
menu.add_selector('Player :', [('Maximilian', 1), ('Maximiliane', 2)], onchange=set_name)
menu.add_selector('Difficulty :', [('Easy', 1), ('Medium', 2), ('Hard', 3)], onchange=set_difficulty)
    # see for more information on the name selector:
    # https://pygame-menu.readthedocs.io/en/3.5.6/_source/widgets_selector.html
menu.add_button('Play', start_the_game)
menu.add_button('Quit', pygame_menu.events.EXIT)
    # see for mor information on the button
    # https://pygame-menu.readthedocs.io/en/3.5.6/_source/widgets_button.html




def redrawGameWindow():
    global mapstartvector #Useful so that everything is aligned nicely within the map.
    #Draw map
    win.fill((0,0,0))
    win.blit(mapnewpic, mapstartvector)

    #Draw Student
    global StudentVec
    #pygame.draw.rect(win, (255,0,0), (StudentVec[0]*30 + mapstartvector[0], StudentVec[1]*30 + mapstartvector[1], 30, 30))
    win.blit(studentmpic, (StudentVec[0]*30 + mapstartvector[0], StudentVec[1]*30 + mapstartvector[1]))

    #Draw Sturm
    global SturmVec
    #pygame.draw.rect(win, (0,255,0), (SturmVec[0]*30 + mapstartvector[0], SturmVec[1]*30 + mapstartvector[1], 30, 30))
    win.blit(sturmpic, (SturmVec[0]*30 + mapstartvector[0], SturmVec[1]*30 + mapstartvector[1]))

    #Draw Token
    global TokenVecs
    pygame.draw.rect(win, (0,0,255), (TokenVec[0]*30 + mapstartvector[0], TokenVec[1]*30 + mapstartvector[1], 30, 30))
    pygame.display.update()

def TakeInput():
    global StudentChange
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == ord('w'):
                StudentChange = np.array([0,-1])
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                StudentChange = np.array([0,1])
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                StudentChange = np.array([-1,0])
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                StudentChange = np.array([1,0])
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_DELETE:
                pygame.event.post(pygame.event.Event(pygame.QUIT)) 

def MoveStudent():
    ## First check if the move is legal -> WITHIN BOUNDS, NOT ON INVALID SPACE, NOT ON STURM
    global StudentVec
    global StudentChange
    global mapwidth
    global mapheight
    global map
    CanMove = True

    #Is the movement within the bounds of the map? (a 30x20 grid)
    nextx = StudentVec[0] + StudentChange[0] 
    nexty = StudentVec[1] + StudentChange[1]

    if nextx < 0:
        CanMove = False
    if nextx > mapwidth - 1:
        CanMove = False
    if nexty < 0:
        CanMove = False
    if nexty > mapheight - 1:
        CanMove = False
    
    #Is the movement to a valid space? (if statement needed to make sure indexing is done within the range of the map)
    if (nextx >= 0 and nexty >= 0 and nextx <= mapwidth - 1 and nexty <= mapheight - 1):
         if map[nexty][nextx] == "0":
            CanMove = False
    
    #Is the movement to Sturm? - Useful failsafe
    if nextx == SturmVec[0] and nexty == SturmVec[1]:
        CanMove = False
    #CanMove = False

    if CanMove: 
        StudentVec = StudentVec + StudentChange

def MoveSturm():
    # Nothing so far
    return True

def TokenHandler():
    # If you catch the toxen, then a new token needs to be initialized - as well as changing the picture Token is using - and changing score
    global score
    
    if StudentVec[0] == TokenVec[0] and StudentVec[1] == TokenVec[1]:
        score = score + 1
        ShuffleToken()

    #In case the token spawns at the same place as before, we want it to go somewhere else instead.
    while StudentVec[0] == TokenVec[0] and StudentVec[1] == TokenVec[1]:
        ShuffleToken()

def ShuffleToken():
    #Choose one of the vectors for the new Token at random
    global TokenVec
    global ListOfTokenSpawnPoints

    randn = random.randint(0, len(ListOfTokenSpawnPoints) - 1)
    TokenVec = ListOfTokenSpawnPoints[randn]

def IsCaught():
    global GameOver
    if np.linalg.norm(StudentVec - SturmVec) <= 1.3:
        GameOver = True
        return True
    else:
        return False

def GameLoop():
    while run:
        pygame.time.delay(100)
        global IsMenuScreen

        if IsMenuScreen == True:
            menu.mainloop(win)
        elif GameOver == False:
            TakeInput()
            MoveStudent()
            MoveSturm()
            TokenHandler()
            IsCaught()
            redrawGameWindow()
        else:
            TakeInput()
            pygame.display.update()

    pygame.quit() 


"""
From here on out, the program is actually run
"""

ShuffleToken()
GameLoop()
    

