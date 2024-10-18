import pygame
from pygame.locals import *

# Define colors for reuse
BG_COLOR = (30, 30, 30)  # Dark background for contrast
GRID_COLOR = (160, 160, 160)  # Light grid lines for visibility on dark background
BOARD_COLOR = (70, 50, 20)  # Darker brown for the board
PLAYER_COLORS = [(20, 20, 20), (230, 230, 230)]  # Dark Black and Off-White stones
OUTLINE_COLOR = (100, 100, 100)
LABEL_COLOR = (200, 200, 200)  # Light labels for better readability

class Board:
    """Class representing the game board for displaying a grid and managing chess pieces."""

    def __init__(self, board_len=15):
        """
        Initialize the board with the given length.

        Parameters
        ----------
        board_len : int
            The length of the board (number of rows and columns).
        """
        self.len = board_len
        self.last_move = None  # Store the last move for highlighting
        self.move_counter = 0  # Count the moves to display numbers on stones
        self.game_over = False  # Game state to control the "New Game" button display

        pygame.init()
        self.clock = pygame.time.Clock()

        # Set up board dimensions and grid size
        self.board_width = self.len * 50 + 100
        self.board_height = self.len * 50 + 100
        self.grid_size = 50

        # Create the display window and set its title
        self.screen = pygame.display.set_mode((self.board_width, self.board_height), pygame.RESIZABLE)
        pygame.display.set_caption('AlphaGobang-lxy-SCUT')

        # Button attributes
        self.button_font = pygame.font.SysFont('Arial', 24)
        self.new_game_button = pygame.Rect(self.board_width // 2 - 160, self.board_height - 60, 150, 40)
        self.exit_button = pygame.Rect(self.board_width // 2 + 10, self.board_height - 60, 150, 40)

        # Draw the initial board
        self.new_game()

    def new_game(self):
        """Reset the board for a new game, drawing the background and grid."""
        self.screen.fill(BG_COLOR)
        pygame.draw.rect(self.screen, BOARD_COLOR, (50, 50, self.len * 50, self.len * 50), 0)
        self.move_counter = 0
        self.last_move = None
        self.game_over = False

        # Draw the grid lines and coordinate labels
        for i in range(self.len):
            x = 75 + i * self.grid_size
            y = 75 + i * self.grid_size
            pygame.draw.line(self.screen, GRID_COLOR, (x, 50), (x, self.board_height - 50), 1)
            pygame.draw.line(self.screen, GRID_COLOR, (50, y), (self.board_width - 50, y), 1)
            self._draw_label(x, 50 - 20, str(i + 1))
            self._draw_label(50 - 20, y, chr(65 + i))

        pygame.display.update()

    def _draw_label(self, x, y, text):
        """Draw a text label at a given position."""
        label = self.button_font.render(text, True, LABEL_COLOR)
        self.screen.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))

    def update(self):
        """Refresh the display to show any changes."""
        pygame.display.update()

    def quit(self):
        """Handle quitting the game when the window is closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def draw_circle(self, pos, player):
        """
        Draw a circle representing a player's move, along with the move number.

        Parameters
        ----------
        pos : tuple of (int, int)
            The position on the board to draw the circle (x, y).

        player : int
            The player number, 0 for black and 1 for white.
        """
        color = PLAYER_COLORS[player]
        pygame.draw.circle(self.screen, color, pos, 22)
        pygame.draw.circle(self.screen, OUTLINE_COLOR, pos, 23, 2)  # Outline for better contrast

        # Draw move number on top of the stone
        self.move_counter += 1
        self._draw_move_number(pos, self.move_counter, player)

    def _draw_move_number(self, pos, move_number, player):
        """Draw the move number on a stone."""
        font = pygame.font.SysFont('Arial', 18)
        text_color = PLAYER_COLORS[1 - player]  # Use contrasting color for visibility
        text = font.render(str(move_number), True, text_color)
        text_rect = text.get_rect(center=pos)
        self.screen.blit(text, text_rect)

    def clear(self):
        """Clear all events from the event queue."""
        pygame.event.clear()

    def get_click(self):
        """
        Wait for a mouse click and return the position.

        Returns
        -------
        a, b : int
            The x and y coordinates of the mouse click.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return event.pos

    def change_pos(self, x, y):
        """
        Convert screen coordinates to board coordinates.

        Parameters
        ----------
        x : int
            The x-coordinate from the screen.

        y : int
            The y-coordinate from the screen.

        Returns
        -------
        x, y : int
            The adjusted board coordinates.
        """
        x = int(x / 50) + (1 if x % 50 >= 25 else 0)
        y = int(y / 50) + (1 if y % 50 >= 25 else 0)
        return x, y

    def display_message(self, message):
        """
        Display a message in the center of the screen, and show a button for starting a new game.

        Parameters
        ----------
        message : str
            The message to display.
        """
        text = self.button_font.render(message, True, (255, 255, 255))
        rect = text.get_rect(center=(self.board_width // 2, self.board_height // 2))
        self.screen.blit(text, rect)

        # Draw the "New Game" and "Exit" buttons
        pygame.draw.rect(self.screen, (100, 200, 100), self.new_game_button)
        new_game_text = self.button_font.render("New Game", True, (0, 0, 0))
        self.screen.blit(new_game_text, new_game_text.get_rect(center=self.new_game_button.center))

        pygame.draw.rect(self.screen, (200, 100, 100), self.exit_button)
        exit_text = self.button_font.render("Exit", True, (0, 0, 0))
        self.screen.blit(exit_text, exit_text.get_rect(center=self.exit_button.center))

        pygame.display.update()
        self.game_over = True

    def handle_events(self):
        """Handle events, including checking for button clicks."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
                    if self.new_game_button.collidepoint(event.pos):
                        self.new_game()
                        return  # Exit loop to start a new game
                    elif self.exit_button.collidepoint(event.pos):
                        pygame.quit()