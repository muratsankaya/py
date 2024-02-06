from tictactoe import TicTacToe
import math
import random
import time

def three_mark_search(board, i, mark):
    div = i // 3 

    # horizontal
    if (i+1)//3 == div and (i+2)//3 == div and board[i+1] == mark and board[i+2] == mark:
        return True
    
    mod = i % 3

    # vertical 
    if i + 6 < len(board) and (i+6)%3 == mod and board[i+3] == mark and board[i+6] == mark:
        return True
    
    # diagonal
    if (i == 0 and board[4] == mark and board[8] == mark) or (i == 2 and board[4] == mark and board[6] == mark):
        return True
    
    return False

def game_over(board) -> None | str:
    """
    Returns:
        - 'O' if 'O' wins
        - 'X' if 'X' wins
        - '-' if tie
        - None if game continues
    """
    mark_count = 0
    
    for i, mark in enumerate(board.board):
        if mark != ' ':
            mark_count += 1
            if(three_mark_search(board.board, i, mark)):
                return mark
            
    if mark_count == len(board.board):
        return '-' # tie
    
    return None
            

# - code assumes the ai_player label will be the maximizer
# - even though I decrement depth in each call, I don't rely on the depth in the base case
#   I let the algorithm calculate till game end.
def minimax(board, depth, maximizing_player, ai_player):

    # human player could also be the other AI
    human_player = 'O' if ai_player == 'X' else 'X'

    go = game_over(board)

    if go is not None:
        return {'position': None, 'score': (1 if go == ai_player else (-1 if go == human_player else 0))}

    if maximizing_player:
        
        best = {'position': None, 'score': -float('inf')}

        for move in board.available_moves():

            board.board[move] = ai_player

            action = minimax(board, depth - 1, False, ai_player)

            # undo move / backtrack
            board.board[move] = ' '

            action['position'] = move

            if action['score'] > best['score']:
                best = action
    
    else: # minimizing

        best = {'position': None, 'score': float('inf')}

        for move in board.available_moves():

            board.board[move] = human_player

            action = minimax(board, depth - 1, True, ai_player)

            # undo move / backtrack
            board.board[move] = ' '

            action['position'] = move

            if action['score'] < best['score']:
                best = action

    return best

    
def minimax_with_alpha_beta(board, depth, alpha, beta, maximizing_player, ai_player):
     # human player could also be the other AI
    human_player = 'O' if ai_player == 'X' else 'X'

    go = game_over(board)

    if go is not None:
        return {'position': None, 'score': (1 if go == ai_player else (-1 if go == human_player else 0))}

    if maximizing_player:
        
        best = {'position': None, 'score': -float('inf')}

        for move in board.available_moves():

            board.board[move] = ai_player

            action = minimax_with_alpha_beta(board, depth - 1, alpha, beta, False, ai_player)

            # undo move / backtrack
            board.board[move] = ' '

            action['position'] = move

            if action['score'] > best['score']:
                best = action

            alpha = max(alpha, action['score'])
            
            if beta <= alpha:
                break
            
    else: # minimizing

        best = {'position': None, 'score': float('inf')}

        for move in board.available_moves():

            board.board[move] = human_player

            action = minimax_with_alpha_beta(board, depth - 1, alpha, beta, True, ai_player)
            
            # undo move / backtrack
            board.board[move] = ' '

            action['position'] = move

            if action['score'] < best['score']:
                best = action

            beta = min(beta, action['score'])
            
            if beta <= alpha:
                break

    return best

def play_game_human_moves_first():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    letter = 'X'  # Human player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # AI's turn
            square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (O) chooses square {square + 1}")
        else:
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")

def play_game_ai_moves_first():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    first_move = True

    letter = 'O'  # AI player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # AI's turn
            if first_move:
                square = random.randint(0, 8)
                first_move = False
            else:
                square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (O) chooses square {square + 1}")
        else:
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")

def play_game_human_vs_human():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    letter = 'O'  # Human (O) player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # Human (O)'s turn
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

                if square is None:
                    print("\nGame is a draw!")
                    break
                game.make_move(square, letter)
                print(f"\nAI (O) chooses square {square + 1}")
        else:
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")

def play_game_ai_vs_ai():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    first_move = True

    letter = 'O'  # AI (O) player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # AI (O)'s turn
            if first_move:
                square = random.randint(0, 8)
                first_move = False
            else:
                square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (O) chooses square {square + 1}")
            time.sleep(0.75)
        else:
            square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (X) chooses square {square + 1}")
            time.sleep(0.75)

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")


if __name__ == '__main__':

    print("""
Modes of play available:

    hh: Hooman vs. hooman
    ha: Hooman vs. AI
    ah: AI vs. Hooman - AI makes first move
    aa: AI vs. AI""")

    valid_move = False
    while not valid_move:
        mode = input("\nEnter preferred mode of play (e.g., aa): ")
        try:
            if mode not in ["hh", "ha", "ah", "aa"]:
                raise ValueError
            valid_move = True
            if mode == "hh":
                play_game_human_vs_human()
            elif mode == "ha":
                play_game_human_moves_first()
            elif mode == "ah":
                play_game_ai_moves_first()
            else:
                play_game_ai_vs_ai()
        except ValueError:
            print("\nInvalid option entered. Try again.")

