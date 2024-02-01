import time
import pyautogui
import keyboard

# Set a delay to give you time to focus on the emulator window
time.sleep(3)

# Press 'j' key
#pyautogui.press('j')
while True:

    interval = 0.016
    interval2 = 0.08
    #interval

    pyautogui.keyDown('s')
    time.sleep(interval) 
    pyautogui.keyDown('d')
    pyautogui.keyUp('s')     
    time.sleep(interval)  
    pyautogui.keyDown('j')
    pyautogui.keyUp('d')
    time.sleep(interval)  
    pyautogui.keyUp('j')

    time.sleep(interval*2) 

    pyautogui.keyDown('a')
    time.sleep(interval*2) 
    pyautogui.keyUp('a') 
    time.sleep(interval*2) 

    pyautogui.keyDown('s')
    time.sleep(interval) 
    pyautogui.keyDown('d')
    pyautogui.keyUp('s')     
    time.sleep(interval2)  
    pyautogui.keyUp('d') 
    pyautogui.keyDown('s')
    time.sleep(interval2) 
    pyautogui.keyDown('d')
    pyautogui.keyUp('s')     
    time.sleep(interval)  
    pyautogui.keyDown('j')
    pyautogui.keyUp('d')
    time.sleep(interval)  
    pyautogui.keyUp('j')











    pyautogui.keyDown('a') 
    pyautogui.keyUp('a')

    time.sleep(interval*5) 
    pyautogui.keyDown('a')    
    time.sleep(1) 
    pyautogui.keyUp('a')

    #keyboard.press('left')
    pyautogui.keyDown('d')
    print("Press 'left d' key")
    time.sleep(1)

    

    pyautogui.keyDown('u')
    time.sleep(0.1) 
    #keyboard.release('left')
    pyautogui.keyUp('d')
    print("release 'left d' key")

    pyautogui.keyUp('u')

    print("Press 'A' key")

    # Press 'k' key after a short delay
    time.sleep(0.5)
    #pyautogui.press('k')
    pyautogui.keyDown('i')
    pyautogui.keyUp('i')
    print("Press 'k' key")

    pyautogui.keyDown('a') 
    time.sleep(interval) 
    pyautogui.keyUp('a')
    time.sleep(interval) 
    pyautogui.keyDown('a') 
    time.sleep(interval) 
    pyautogui.keyUp('a')
    time.sleep(interval) 

    pyautogui.keyDown('s') 
    time.sleep(interval) 
    pyautogui.keyUp('s')
    time.sleep(interval) 
    pyautogui.keyDown('s') 
    time.sleep(interval) 
    pyautogui.keyUp('s')
    time.sleep(interval) 
    
    time.sleep(2)

# Set a delay to give you time to focus on the emulator window
time.sleep(5)

# Press 'A' and 'B' keys simultaneously
keyboard.press('j')
keyboard.press('left')

# Release both keys after a short delay
time.sleep(0.5)
keyboard.release('j')
keyboard.release('left')



# Example of holding down a key for 2 seconds
pyautogui.keyDown('left')
print("Press 'left' key")
time.sleep(0.2)
pyautogui.keyUp('left')
print("Press 'left' key")
