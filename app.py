import streamlit as st
import numpy as np
import torch
from PIL import Image
import os
from othello import OthelloGame, OthelloAI, DQN  # Assuming othello.py contains your game classes

# Initialize session state
if 'game' not in st.session_state:
    st.session_state.game = OthelloGame()
if 'ai' not in st.session_state:
    st.session_state.ai = OthelloAI()
    # Load the pre-trained model
    model_path = 'othello_model.pth'
    if os.path.exists(model_path):
        st.session_state.ai.load_model(model_path)
    else:
        st.warning("Pre-trained model not found. AI will play randomly.")
if 'human_color' not in st.session_state:
    st.session_state.human_color = 1  # 1 for black (first), -1 for white (second)
if 'game_over' not in st.session_state:
    st.session_state.game_over = False

def reset_game():
    st.session_state.game = OthelloGame()
    st.session_state.game_over = False
    if st.session_state.human_color == -1:  # If human is second
        # AI makes first move
        valid_moves = st.session_state.game.get_valid_moves()
        if valid_moves:
            action = st.session_state.ai.get_action(st.session_state.game.get_state(), valid_moves, training=False)
            if action:
                st.session_state.game.make_move(*action)

def create_board_image():
    # Create a grid image with current game state
    cell_size = 60
    board_size = 8
    image = Image.new('RGB', (cell_size * board_size, cell_size * board_size), 'darkgreen')
    
    # Draw grid lines
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Draw lines
    for i in range(board_size + 1):
        draw.line([(i * cell_size, 0), (i * cell_size, cell_size * board_size)], fill='black', width=2)
        draw.line([(0, i * cell_size), (cell_size * board_size, i * cell_size)], fill='black', width=2)
    
    # Draw pieces
    for i in range(board_size):
        for j in range(board_size):
            if st.session_state.game.board[i, j] != 0:
                x = j * cell_size + cell_size // 2
                y = i * cell_size + cell_size // 2
                color = 'black' if st.session_state.game.board[i, j] == 1 else 'white'
                draw.ellipse([(x - 25, y - 25), (x + 25, y + 25)], fill=color)
    
    return image

def handle_click(i, j):
    if st.session_state.game_over:
        return
    
    if st.session_state.game.current_player != st.session_state.human_color:
        st.warning("Not your turn!")
        return
    
    if not st.session_state.game.is_valid_move(i, j):
        st.warning("Invalid move!")
        return
    
    # Make human move
    st.session_state.game.make_move(i, j)
    
    # Check if game is over
    if not st.session_state.game.get_valid_moves():
        st.session_state.game_over = True
        return
    
    # AI's turn
    valid_moves = st.session_state.game.get_valid_moves()
    if valid_moves:
        action = st.session_state.ai.get_action(st.session_state.game.get_state(), valid_moves, training=False)
        if action:
            st.session_state.game.make_move(*action)
    
    # Check if game is over after AI move
    if not st.session_state.game.get_valid_moves():
        st.session_state.game_over = True

def main():
    st.title("Othello vs AI")
    
    # Color selection
    col1, col2 = st.columns([2, 2])
    with col1:
        color = st.radio("Choose your color:", ["Black (First)", "White (Second)"], 
                        index=0 if st.session_state.human_color == 1 else 1,
                        on_change=reset_game)
        st.session_state.human_color = 1 if color == "Black (First)" else -1
    
    with col2:
        if st.button("New Game"):
            reset_game()
    
    # Display current player
    current_player = "Your" if st.session_state.game.current_player == st.session_state.human_color else "AI's"
    st.write(f"{current_player} turn")
    
    # Create and display the board
    board_image = create_board_image()
    
    # Create clickable grid
    clicked = st.image(board_image, use_column_width=True)
    
    # Handle clicks
    if clicked:
        x, y = st.experimental_get_query_params().get('click', [[0, 0]])[0]
        cell_size = board_image.width // 8
        i, j = int(float(y) // cell_size), int(float(x) // cell_size)
        handle_click(i, j)
    
    # Display game status
    if st.session_state.game_over:
        winner = st.session_state.game.get_winner()
        if winner == st.session_state.human_color:
            st.success("You win!")
        elif winner == -st.session_state.human_color:
            st.error("AI wins!")
        else:
            st.info("It's a tie!")
    
    # Display score
    black_count = np.sum(st.session_state.game.board == 1)
    white_count = np.sum(st.session_state.game.board == -1)
    st.write(f"Score - Black: {black_count}, White: {white_count}")

if __name__ == "__main__":
    main()
