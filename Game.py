import pygame
import pygame_menu
import numpy as np
import random
import pickle
import neat
import os
from mapfile import *

from NEAT_Implementation import GiveInputStudent
#NEAT Stuff
StudentAIPowered = True
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    with open("winner.pkl", "rb") as f:
        genome = pickle.load(f)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
### INITIALIZING VARIABLES

pygame.init()


### Graphics

mappic = pygame.image.load("pics/HSG_Map_Outline.jpg")
mapnewpic = pygame.image.load("pics/MapGrass_Progress_Street.png")
mapstartvector = np.array([0, 0])
sturmpic = pygame.image.load("pics/Sturm30.png")
studentpic = pygame.image.load("pics/man.png")
studentmpic = pygame.image.load("pics/man.png")
studentfpic = pygame.image.load("pics/woman.png")
map3 = [
    "000000000000000000000000000000",    
    "000000000010000000000000000000",
    "000000000010000000000011000000",
    "000000000010000000001110000000",
    "000000001111111111111000000000",
    "000000001111111000010000000000",
    "000001111100011000011100000000",        
    "000001111100011000000100000000",   
    "000001110011111111000111111110",
    "000001110011111111000100000000",                         
    "000001110011111111111100000000",                    
    "000001111111111111111100000000",
    "000000001110000011001100000000",
    "000000001110000011001100000000",
    "000000001110000001001100000000",
    "000000001110000001001100000000", 
    "000000000100000001111100000000",              
    "000000000100000001000100000000",   
    "000000000100000001000100000000",
    "000000000111111111111111111110"                  
]
map2 = [
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111",
    "111111111111111111111111111111"
]
map = [
    "000000000000000000000000000000",    
    "000000000010000000000000000000",
    "000000000010000000000011000000",
    "000000000010000000001110000000",
    "000000001111111111111000000000",
    "000000001111111000010000000000",
    "000001111110001000011100000000",        
    "000001111110001000000100000000",   
    "000001110011111111000111111110",
    "000001110011111111000100000000",                         
    "000001110011111111111100000000",                    
    "000001111111111111111100000000",
    "000000001110000011001100000000",
    "000000001110000011001100000000",
    "000000001110000001001100000000",
    "000000001110000001001100000000", 
    "000000000100000001111100000000",              
    "000000000100000001000100000000",   
    "000000000100000001000100000000",
    "000000000111111111111111111110"                  
]
win = pygame.display.set_mode((900, 600))
pygame.display.set_caption("HSG Heist")


#Game related variable
FPS = 10
mapwidth = 30
mapheight = 20
run = True
score = 0
GameOver = False
IsMenuScreen = True

# Student stuff
StudentVec = np.array([13,9])
StudentChange = np.array([1,0])

# Sturm stuff
SturmVec = np.array([0,19])
SturmChange = np.array([-1,0])

# Token stuff
TokenVec = np.array([3,0])
ListOfTokenSpawnPoints = [np.array([10,2]),np.array([12,4]),np.array([21,6]),np.array([5,6]),np.array([5,11]), np.array([11,8]), np.array([16,8]), np.array([25,8]), np.array([10,9]), np.array([16,13]), np.array([19,11]), np.array([24,19])]
#[[10,2],[12,4],[21,6],[5,6],[5,11], [11,8], [16,8], [25,8], [10,9], [16,13], [19,11], [24,19]]

"""def set_difficulty(value, difficulty):
    # insert code here
    pass"""     

def start_the_game():
    global IsMenuScreen
    IsMenuScreen = False
    ShuffleToken()
    GameLoop()
   
def IsMenuScreenTrue():
    global IsMenuScreen
    return IsMenuScreen

def set_name(value, name):
    global studentpic
    print(value)
    if value[0][1] == 1:
        studentpic = studentmpic
    if value[0][1] == 2:
        studentpic = studentfpic
    return True
    
menu = pygame_menu.Menu(600, 900, 'Startwoche 2021 – Introduction Game!',
                       theme=pygame_menu.themes.THEME_GREEN)
    # I am not sure if THEME_GREEN exists, since I cannot get it to run on my machine. Default Theme is "BLUE"
menu.add_text_input('This game will familiarize you with the premises of the HSG,')
menu.add_text_input('without needing to visit the university in person.')
menu.add_text_input('Get all the stuff before Sturm catches you!')
    # documentation on text inserts
    # https://pygame-menu.readthedocs.io/en/3.5.6/_source/widgets_textinput.html
menu.add_selector('Player :', [('Maximilian', 1), ('Maximiliane', 2)], onchange=set_name)
menu.add_button('Play', start_the_game)
menu.add_button('Quit', pygame_menu.events.EXIT)
 


def GameLoop():
    BestStudentNet = neat.nn.FeedForwardNetwork.create(genome, config)
    while run:
        global FPS
        pygame.time.delay(FPS)
        global IsMenuScreen
        global StudentChange
        if GameOver == False:
            
            InputVec = GiveInputStudent(StudentVec, SturmVec, TokenVec)
            StudentChange = BestStudentNet.activate(InputVec)
            TakeInput()
            print("whats this")
            print(StudentChange)
            #Normalize NN Output - to which basis vector is (x,y) leaning the most?
            if(abs(StudentChange[0]) >= abs(StudentChange[1])):
                StudentChange[1] = 0
                if(StudentChange[0] > 0):
                    StudentChange[0] = 1
                else:
                    StudentChange[0] = -1
            else:
                StudentChange[0] = 0
                if(StudentChange[1] > 0):
                    StudentChange[1] = 1
                else:
                    StudentChange[1] = -1
            
            print(StudentChange)
            MoveStudent()
            MoveSturm()
            TokenHandler()
            IsCaught()
            redrawGameWindow()
        else:
            TakeInput()
            pygame.display.update()

    pygame.quit() 

def redrawGameWindow():
    global mapstartvector #Useful so that everything is aligned nicely within the map.
    #Draw map
    win.fill((0,0,0))
    win.blit(mapnewpic, mapstartvector)

    #Draw Student
    global StudentVec
    #pygame.draw.rect(win, (255,0,0), (StudentVec[0]*30 + mapstartvector[0], StudentVec[1]*30 + mapstartvector[1], 30, 30))
    win.blit(studentpic, (StudentVec[0]*30 + mapstartvector[0], StudentVec[1]*30 + mapstartvector[1]))

    #Draw Sturm
    global SturmVec
    #pygame.draw.rect(win, (0,255,0), (SturmVec[0]*30 + mapstartvector[0], SturmVec[1]*30 + mapstartvector[1], 30, 30))
    win.blit(sturmpic, (SturmVec[0]*30 + mapstartvector[0], SturmVec[1]*30 + mapstartvector[1]))

    #Draw Token
    global TokenVec
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

def IsValid(Vec):
    global map
    global mapheight
    global mapwidth
    if Vec[0] < 0:
        return False
    if Vec[0] > mapwidth - 1:
        return False
    if Vec[1] < 0:
        return False
    if Vec[1] > mapheight - 1:
        return False
    
    #Is the movement to a valid space? (if statement needed to make sure indexing is done within the range of the map)
    if map[Vec[1]][Vec[0]] == "0":
            return False
    else:
        return True

def MoveStudent():
    ## First check if the move is legal -> WITHIN BOUNDS, NOT ON INVALID SPACE, NOT ON STURM
    global StudentVec
    global StudentChange
    global mapwidth
    global mapheight
    global map

    #Is the movement within the bounds of the map? (a 30x20 grid)
    NextVec = StudentVec + StudentChange
    #If you move into a wall, move randomly - otherwise the AI will stay stuck.
    while not(IsValid(NextVec)):
        #Is this movement legal?
        

        #If you hit a wall, move randomly to a valid space
        PossibleMovementList = []
        BasisVecList = [[1,0], [0,1], [-1,0], [0,-1]]
        for i in range(4):
            if IsValid(BasisVecList[i] + StudentVec):
                PossibleMovementList.append(BasisVecList[i])
        
        Rand = random.randint(0, len(PossibleMovementList) - 1)
        StudentChange = PossibleMovementList[Rand]
        NextVec = StudentVec + StudentChange

    #Is the movement to Sturm? - Useful failsafe
    if not(NextVec[0] == SturmVec[0] and NextVec[1] == SturmVec[1]):
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


"""
From here on out, the program is actually run
"""
menu.mainloop(win)

    

