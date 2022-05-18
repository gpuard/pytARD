import os
import platform

class Notification:
    '''
    Notifies the user of action the software is currently doing via desktop notification. Supports Linux and macOS.
    '''

    def __init__(self):
        '''
        No implementation
        '''
        pass

    @staticmethod
    def notify(title: str, message: str):
        '''
        Notifies the user of action the software is currently doing via desktop notification.

        Parameters
        ----------
        title : str
            Title of the notification
        message : str
            Message of the notification

        '''
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