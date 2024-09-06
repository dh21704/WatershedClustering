import random

# Player and enemy definitions
player = {
    "name": "Main Player",
    "max health": 20,
    "health": 20,
    "attack": 2,
    "level": 1,
    "xp": 0
}

enemy_1 = {
    "name": "Enemy 1",
    "description": "This guy is super duper evil.",
    "health": 6,
    "attack": 2,
}

enemy_2 = {
    "name": "Enemy 2",
    "description": "Man I hate this guy.",
    "health": 6,
    "attack": 2,
}

enemies = [enemy_1, enemy_2]

def get_enemy():
    return random.choice(enemies)

def print_status(player, enemy):
    print(f"{player['name']} has {player['health']} health")
    print(f"{enemy['name']} has {enemy['health']} health")

def player_attack(player, enemy):
    enemy['health'] -= player['attack']
    print(f"You attack {enemy['name']} for {player['attack']} damage!")
    if enemy['health'] <= 0:
        print(f"You defeated {enemy['name']}")
        player['xp'] += 1  # Increment XP for defeating an enemy
        return True
    return False

def enemy_attack(player, enemy):
    player['health'] -= enemy['attack']
    print(f"{enemy['name']} attacks you for {enemy['attack']} damage!")
    if player['health'] <= 0:
        print("You have been defeated!")
        return True
    return False

def fight(player, enemy):
    print(f"A wild {enemy['name']} appears!")
    print(enemy["description"])
    print("\n------------- BATTLE START! --------------\n")

    while player['health'] > 0 and enemy['health'] > 0:
        print_status(player, enemy)
        print("1. Attack")
        print("2. Run")
        choice = input("What will you do? ")
        print()  # for formatting

        if choice == "1":
            if player_attack(player, enemy):
                break
            if enemy_attack(player, enemy):
                break
        elif choice == "2":
            print("You run away...")
            break
        else:
            print("Invalid choice. Try again.")

    print("\n------------- BATTLE END! --------------\n")
    print(f"Post-Battle Status: Health: {player['health']}, XP: {player['xp']}")

def level_up(player):
    player['level'] += 1
    player['max health'] += 5
    player['health'] = player['max health']
    player['attack'] += 1
    player['xp'] = 0
    print(f"You leveled up! You are now level {player['level']}!")

def game_over(player):
    print(f"Game Over! {player['name']} reached level {player['level']}.")
    exit()

while True:
    enemy = get_enemy()
    fight(player, enemy)
    
    # Check for game-over conditions
    if player['health'] <= 0:
        game_over(player)
    
    # Check if player has enough XP to level up
    if player['xp'] >= player['level'] * 3:
        level_up(player)
    
    # Optional: Print player's status after leveling up or before next fight
    print(f"Current Status: Health: {player['health']}, XP: {player['xp']}, Level: {player['level']}")
