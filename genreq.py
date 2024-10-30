from pip._internal.operations.freeze import freeze

with open('requirement.txt', mode='w') as file:
    for i in freeze(local_only=True):
        file.write(f"{i}\n")