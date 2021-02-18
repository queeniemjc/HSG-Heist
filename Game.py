import pygame
import pygame_menu
import numpy as np
import random
import pickle
import neat
import os
from NEAT_Implementation import GiveInput

#NEAT Stuff
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
mappic = pygame.image.load("pics/Map_NoScore.png")
mapstartvector = np.array([0, 0])
sturmpic = pygame.image.load("pics/Sturm30.png")
studentpic = pygame.image.load("pics/man.png")
studentmpic = pygame.image.load("pics/man.png")
studentfpic = pygame.image.load("pics/woman.png")
Token = []
Token.append(pygame.image.load("pics/Token/beer_30.png"))
Token.append(pygame.image.load("pics/Token/donut_30.png"))
Token.append(pygame.image.load("pics/Token/gym_30.png"))
Token.append(pygame.image.load("pics/Token/map_30.png"))
Token.append(pygame.image.load("pics/Token/pen_30.png"))
Token.append(pygame.image.load("pics/Token/pizza_30.png"))
Token.append(pygame.image.load("pics/Token/sweets_30.png"))
Token.append(pygame.image.load("pics/Token/wine_30.png"))
win = pygame.display.set_mode((900, 600))
pygame.display.set_caption("Startweek Introduction: #EHA")

### Sound
win_token = pygame.mixer.Sound("sound/win_token.wav")
you_lost = pygame.mixer.Sound("sound/you_lost.wav")
hsgsong = pygame.mixer.music.load("sound/hsg_song.mp3")

# Maps to play the game with

validmap = [
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
GameMap = [
    "000000000000000000000000000000",    
    "000000000010000000000000000000",
    "000000000010000000000000000000",
    "000000000010000000000000000000",
    "000000001111111111111111111110",
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



#Game related variable
GameSpeed = 100
mapwidth = 30
mapheight = 20
Score = 0
GameOver = False

# Student stuff
StudentVec = np.array([13,9])
StudentChange = np.array([-1,0])

# Sturm stuff
SturmVec = np.array([17,19])
SturmChange = np.array([1,0])

# Token stuff
TokenVec = np.array([3,0])
ListOfTokenSpawnPoints = [np.array([10,2]),np.array([12,4]),np.array([21,6]),np.array([5,6]),np.array([5,11]), np.array([11,8]), np.array([16,8]), np.array([25,8]), np.array([10,9]), np.array([16,13]), np.array([19,11]), np.array([24,19])]



def start_the_game():
    pygame.mixer.music.play(-1)
    ShuffleToken()
    GameLoop()

def set_difficulty(value, name):
    global GameSpeed
    if value[0][1] == 0:
        GameSpeed = 200
    if value[0][1] == 1:
        GameSpeed = 150
    if value[0][1] == 2:
        GameSpeed = 100

def set_name(value, name):
    global studentpic
    if value[0][1] == 1:
        studentpic = studentmpic
    if value[0][1] == 2:
        studentpic = studentfpic
    return True
    
menu = pygame_menu.Menu(600, 900, 'Startwoche 2021 – Introduction Game!',
                       theme=pygame_menu.themes.THEME_GREEN)
menu.add_text_input('This game will familiarize you with the premises of the HSG')
menu.add_text_input('without needing to visit the university in person!')
menu.add_text_input('Get all the goodies before the prof catches you!')
menu.add_selector('Player :', [('Maximilian', 1), ('Maximiliane', 2)], onchange=set_name)
menu.add_selector('Difficulty :', [('Easy', 0), ('Medium', 1), ('Hard', 2)], onchange=set_difficulty)
menu.add_button('Play', start_the_game)
menu.add_button('Quit', pygame_menu.events.EXIT)
 


def GameLoop():
    #Sturm AIs NN is loaded in
    SturmNet = neat.nn.FeedForwardNetwork.create(genome, config)
    global StudentChange
    global StudentVec
    global SturmVec
    global Score
    global GameOver
    while not(GameOver):
        global GameSpeed
        pygame.time.delay(GameSpeed)

        if not GameOver:
            #Movement controls of Student - inputted by user
            TakeInput()

            #Movement controls of Sturm
            SturmInputVec = GiveInput(SturmVec, TokenVec, StudentVec)
            SturmChange = SturmNet.activate(SturmInputVec)
            
            
            #Normalize NN Output - to which basis vector is (x,y) leaning the most?
            StudentChange = DiscretizeNNOutput(StudentChange)
            SturmChange = DiscretizeNNOutput(SturmChange)
            #Apply this change if possible
            StudentVec = Move(StudentVec, StudentChange, SturmVec)
            SturmVec = Move(SturmVec, SturmChange, StudentVec)
            #Update game
            TokenHandler()
            IsCaught()

            #Output
            RedrawGameWindow()

    pygame.mixer.music.load("sound/win_music.mp3")
    pygame.mixer.music.play(-1)
    while GameOver:
        font = pygame.font.Font(None, 36)
        text = font.render("Game Over! You failed assessment.", True, (255,255,255))
        text_rect = text.get_rect()
        text_x = win.get_width() / 2 - text_rect.width / 2
        text_y = win.get_height() / 2 - text_rect.height / 2
        win.blit(text, [text_x, text_y])
        pygame.time.delay(GameSpeed)
        pygame.display.update()
        TakeInput()
        
   # pygame.quit() 

def DiscretizeNNOutput(StudentChange):
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
    return StudentChange

def RedrawGameWindow():
    global mapstartvector #Useful so that everything is aligned nicely within the map.
    #Draw map
    win.fill((0,0,0))
    win.blit(mappic, mapstartvector)

    #Draw Student
    global StudentVec
    win.blit(studentpic, (StudentVec[0]*30 + mapstartvector[0], StudentVec[1]*30 + mapstartvector[1]))

    #Draw Sturm
    global SturmVec
    win.blit(sturmpic, (SturmVec[0]*30 + mapstartvector[0], SturmVec[1]*30 + mapstartvector[1]))

    #Draw Token
    global TokenVec

    win.blit(Token[Score % len(Token)], (TokenVec[0]*30 + mapstartvector[0], TokenVec[1]*30 + mapstartvector[1]))
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
    global GameMap
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
    if GameMap[Vec[1]][Vec[0]] == "0":
            return False
    else:
        return True

def Move(Vec, Change, OpponentVec):
    ## First check if the move is legal -> WITHIN BOUNDS, NOT ON INVALID SPACE, NOT ON STURM
    global mapwidth
    global mapheight
    global GameMap
    CanMove = True
    #Is the movement within the bounds of the map? (a 30x20 grid)
    NextVec = Vec + Change
    CanMove = IsValid(NextVec)        
            
    #Is the movement to Sturm? - Useful failsafe
    if NextVec[0] == OpponentVec[0] and NextVec[1] == OpponentVec[1]:
        CanMove = False
    if CanMove: 
        Vec[0] = Vec[0] + Change[0]
        Vec[1] = Vec[1] + Change[1]
    return Vec

def TokenHandler():
    # If you catch the toxen, then a new token needs to be initialized - as well as changing the picture Token is using - and changing Score
    global Score
    
    if StudentVec[0] == TokenVec[0] and StudentVec[1] == TokenVec[1]:
        Score = Score + 1
        win_token.play()
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

    

