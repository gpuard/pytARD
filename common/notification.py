import os
import platform

class Notification:

    def __init__():
        pass

    @staticmethod
    def notify(title, message):
        plt = platform.system()
        if plt=='Darwin':
            command = f'''
            osascript -e 'display notification "{message}" with title "{title}"'
            '''
        if plt=='Linux':
            command = f'''
            notify-send "{title}" "{message}"
            '''
        else:
            return
        os.system(command)