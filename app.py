
import streamlit as st
import numpy as np
import torch
from collections import deque
import requests
import os
import io

# [Previous class definitions for OthelloGame, DQN, DQN_Agent, and OthelloAI go here]

# Initialize session state
if 'game' not in st.session_state:
    st.session_state.game = OthelloGame()
    st.session_state.ai = OthelloAI()
    # Load the model from Hugging Face
    try:
        st.session_state.ai.load_model("https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth")
        st.success("AI model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Streamlit interface
st.title("Play Othello Against AI")

# Player selection
player_color = st.radio("Choose your color:", ["Black (First)", "White (Second)"], key="player_color")
human_first = player_color == "Black (First)"

# Reset game button
if st.button("Reset Game"):
    st.session_state.game = OthelloGame()
    if not human_first:  # If AI goes first
        valid_moves = st.session_state.game.get_valid_moves()
        if valid_moves:
            action = st.session_state.ai.ai.get_action(
                st.session_state.game.get_state(),
                valid_moves,
                training=False
            )
            if action:
                st.session_state.game.make_move(*action)

# Create a compact game board display
cols = st.columns([1, 2, 1])  # Create three columns for layout
with cols[1]:  # Use the middle column for the board
    # Create a grid of buttons for the board
    board = st.session_state.game.get_state()
    for i in range(8):
        cols_board = st.columns(8)
        for j in range(8):
            with cols_board[j]:
                # Determine the piece to display
                if board[i][j] == 1:
                    piece = "⚫"
                elif board[i][j] == -1:
                    piece = "⚪"
                else:
                    piece = "·"
                
                # Create a button for each cell
                if st.button(piece, key=f"cell_{i}_{j}"):
                    # Check if it's the player's turn
                    current_player = st.session_state.game.current_player
                    if (human_first and current_player == 1) or \
                       (not human_first and current_player == -1):
                        # Make the move if valid
                        if st.session_state.game.is_valid_move(i, j):
                            st.session_state.game.make_move(i, j)
                            
                            # AI's turn
                            valid_moves = st.session_state.game.get_valid_moves()
                            if valid_moves:
                                action = st.session_state.ai.ai.get_action(
                                    st.session_state.game.get_state(),
                                    valid_moves,
                                    training=False
                                )
                                if action:
                                    st.session_state.game.make_move(*action)

# Game status
winner = st.session_state.game.get_winner()
if winner is not None:
    if winner == 1:
        st.success("Black wins!")
    elif winner == -1:
        st.success("White wins!")
    else:
        st.info("It's a tie!")

# Show valid moves
valid_moves = st.session_state.game.get_valid_moves()
if valid_moves:
    st.write("Valid moves:", valid_moves)
else:
    if winner is None:
        st.write("No valid moves available. Turn passes.")

# Current player indicator
current_player = "Black" if st.session_state.game.current_player == 1 else "White"
st.write(f"Current player: {current_player}")

# Add a scoreboard
board = st.session_state.game.get_state()
black_count = np.sum(board == 1)
white_count = np.sum(board == -1)
st.write(f"Score - Black: {black_count}, White: {white_count}")




