import os
import subprocess


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEUE_TXT = os.path.join(BASE_DIR, "training_queue.txt")
STATUS_FILE = os.path.join(BASE_DIR, "tmp/status.txt")


def main_window():
    subprocess.Popen([
        "gnome-terminal", "--",
        "bash", "-c", f"watch -n 1 cat {STATUS_FILE}; exec bash"   
    ])


def update_status(index, status):
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    base_line = lines[index].split(" | ")[0]
    lines[index] = f"{base_line} | {status}\n"

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


def start_new_process(cmd):
    process = subprocess.Popen([
        "gnome-terminal", "--",
        "bash", "-c", f"{cmd}; exec bash"   
    ])

    return process
    

def read_txt(txt_file):
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.readlines()
    except Exception as e:
        print(f"[ERROR] Не удалось открыть txt-файл: {e}")
        content = []
    
    return content


def process_line(line):
    try:
        arguments = line.strip().split()
        
        if not arguments[0].startswith("python3"):
            arguments.insert(0, "python3")
        
        if not arguments[1].endswith(".py"):
            arguments[1] += ".py"
        
        return " ".join(arguments)
    except Exception as e:
        print(f"[ERROR] Ошибка при обработке команды: {e}")
        return None


def main():
    content = read_txt(QUEUE_TXT)
    
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        for line in content:
            f.write(line + " | Ждет выполнения\n")

    main_window()
    
    commands = []
    for line in content:
        command = process_line(line)
        commands.append(command)

    for i, command in enumerate(commands):
        try:
            update_status(i, "Выполняется")
            process = start_new_process(command)
            process.wait()
            update_status(i, "Выполнено")
        except Exception as e:
            update_status(i, f"Ошибка: {e}")

if __name__ == "__main__":
    main()