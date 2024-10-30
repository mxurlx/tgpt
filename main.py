import logging
import json
import asyncio
import aiosqlite
import os
import torch
from addons import parser
from functools import wraps
from time import time
from ollama import AsyncClient
from diffusers import FluxPipeline

import tools

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ForceReply
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, CallbackQueryHandler, Application, ConversationHandler
from telegram.error import BadRequest


import random
import aiofiles

with open('config.json', mode='r') as file:
    config = json.load(file)

def message(role: str, content: str):
    return {
        'role': role,
        'content': content
    }
    
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

LAST_COMMAND = {}

HAS_INIT = False

M_LANG = config['M_LANG']

CHAT, NEWCHAT, GENERATEIMAGE, ASPECTRATIO, FILECHAT, DELETECHAT, CHATTITLE, CHATHISTORY, SELECTCHAT = range(9)

SYSTEM = message('main', config['SYSTEM'])

ratios = {}
for i in enumerate(tools.ratios.items(), 1):
    ratios[i[0]] = i[1][0]

class Database:
    def __init__(self):
        self.cursor = None
        self.conn = None
    
    async def close(self):
        if self.conn:
            await self.conn.close()

    async def connect(self):     
        self.conn = await aiosqlite.connect('chatdb.db')
        self.cursor = await self.conn.cursor()
        await self.cursor.execute('''CREATE TABLE IF NOT EXISTS chats
                                  (cid INTEGER NOT NULL, chat_id INTEGER NOT NULL, history_path TEXT NOT NULL,
                                  FOREIGN KEY (chat_id) REFERENCES settings(chat_id))''')
        await self.cursor.execute('''CREATE TABLE IF NOT EXISTS settings
                                  (chat_id INTEGER PRIMARY KEY, language TEXT, current_chat INTEGER, aspect_ratio TEXT)
                                  ''')
        await self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_chats_id ON settings(chat_id)")
        await self.conn.commit()
        
    async def new_chat(self, chat_id: int):
        await self.cursor.execute("SELECT * FROM settings WHERE chat_id = ?", (chat_id,))
        row = await self.cursor.fetchone()
        if not row:
            await self.cursor.execute("INSERT OR REPLACE INTO settings (chat_id) VALUES (?)", (chat_id,))
            await self.conn.commit()
    
    async def get_language(self, chat_id: int):
        await self.cursor.execute("SELECT language FROM settings WHERE chat_id = ?", (chat_id,))
        row = await self.cursor.fetchone()
        return row[0] if row else None
    
    async def set_language(self, chat_id: int, val):
        await self.new_chat(chat_id)
        await self.cursor.execute("UPDATE settings SET language = ? WHERE chat_id = ?", (val, chat_id))
        await self.conn.commit()
    
    async def get_histories(self, chat_id: int):
        await self.cursor.execute("SELECT cid, history_path FROM chats JOIN settings ON chats.chat_id = settings.chat_id WHERE chats.chat_id = ? ORDER BY chats.cid", (chat_id,))
        rows = await self.cursor.fetchall()
        return rows if rows else None
    
    async def get_history(self, cid: int, chat_id: int):
        await self.cursor.execute("SELECT cid, history_path FROM chats JOIN settings ON chats.chat_id = settings.chat_id WHERE chats.cid = ? AND chats.chat_id = ?", (cid, chat_id,))
        row = await self.cursor.fetchone()
        return row if row else None
    
    async def set_history(self, cid: int, chat_id: int, val):
        await self.cursor.execute("INSERT OR IGNORE INTO chats (cid, chat_id, history_path) VALUES (?, ?, ?)", (cid, chat_id, val,))
        await self.conn.commit()
    
    async def delete_history(self, cid: int, chat_id: int):
        await self.cursor.execute("DELETE FROM chats WHERE cid = ? AND chat_id = ?", (cid, chat_id,))
        await self.conn.commit()
    
    async def get_current_chat(self, chat_id: int):
        await self.cursor.execute("SELECT current_chat FROM settings WHERE chat_id = ?", (chat_id,))
        row = await self.cursor.fetchone()
        return row[0] if row else None
    
    async def set_current_chat(self, chat_id: int, val):
        await self.new_chat(chat_id)
        await self.cursor.execute("UPDATE settings SET current_chat = ? WHERE chat_id = ?", (val, chat_id))
        await self.conn.commit()
    
    async def get_aspect_ratio(self, chat_id: int):
        await self.cursor.execute("SELECT aspect_ratio FROM settings WHERE chat_id = ?", (chat_id,))
        row = await self.cursor.fetchone()
        return row[0] if row else None
    
    async def set_aspect_ratio(self, chat_id: int, val):
        await self.new_chat(chat_id)
        await self.cursor.execute("UPDATE settings SET aspect_ratio = ? WHERE chat_id = ?", (val, chat_id))
        await self.conn.commit()

DB = Database()


def command_handler(seconds: int = 0):
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            
            lang = await DB.get_language(update.effective_chat.id)
            if lang is None:
                lang = 'english'

            context.user_data['lang'] = lang
            
            user_id = update.effective_user.id
            
            req = update.message.text if not context.args else ' '.join(context.args)
            
            
            if context.user_data.get('sent') or seconds != 0:
                print('-'*20, f"Request from {user_id}", req, '-'*20, sep='\n')
            
            now = time()
            for id, timest in LAST_COMMAND.items():
                if now < timest:
                    LAST_COMMAND.pop(id)

            if user_id in LAST_COMMAND and now - LAST_COMMAND[user_id] < seconds and context.user_data.get('file') is None:
                return
            LAST_COMMAND[user_id] = now
            
            await asyncio.sleep(config['GENERAL_DELAY'])
            
            return await func(update, context, *args, **kwargs)
        
        return wrapper
    return decorator





@command_handler(config['DELAY'])
async def language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global USER_LANG_REQUEST
    lang = context.user_data.get('lang')
    if not context.args:

        USER_LANG_REQUEST = update.effective_user.id

        keyboard = [
            [InlineKeyboardButton('English', callback_data='english')],
            [InlineKeyboardButton('Русский', callback_data='russian')],
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(M_LANG[lang]['select_language'], reply_markup=reply_markup)

    else:
        await changelanguage(update, context, context.args[0].lower())

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global USER_LANG_REQUEST

    if update.effective_user.id != USER_LANG_REQUEST:
        print(f'\n\nUNAUTHORIZED {update.effective_user.id}\n\n')
        return
    query = update.callback_query
    await query.answer()

    await DB.set_language(update.effective_chat.id, query.data)
    lang = await DB.get_language(update.effective_chat.id)
    await query.edit_message_text(M_LANG[lang]['select_language_complete'])

async def changelanguage(update: Update, context: ContextTypes.DEFAULT_TYPE, newlang: str):
    if M_LANG.get(newlang) is None:
        await update.message.reply_text(M_LANG[lang]['language_not_found'])
        return

    await DB.set_language(update.effective_chat.id, newlang)
    lang = await DB.get_language(update.effective_chat.id)
    await update.message.reply_text(M_LANG[lang]['select_language_complete'])


@command_handler(config['DELAY'])
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang')
    await update.message.reply_text(text=M_LANG[lang]['start'])
    
punct = {
    '***', '**', '*', '```'
}

async def write(update: Update, context: ContextTypes.DEFAULT_TYPE, messages: list, edit: int):
    last_time = time()
    wholeMessage = ""
    chunk = ""
    mark = ""
    sent = None
    async for part in await AsyncClient().chat(model=config['MODEL'], messages=messages, stream=True):
        wholeMessage += part['message']['content']
        chunk += part['message']['content']
        #print(part['message']['content'])
        if part['message']['content'] in punct:
            if part['message']['content'] != mark:
                mark = part['message']['content']
            else:
                mark = ""
        if time() - last_time >= config['STREAM_DELAY']:
            try:
                buf = chunk
                if mark == '```':
                    mark = '\n```\n'
                if mark:
                    buf += mark
                sent = await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=edit,
                    text=buf,
                    parse_mode='Markdown'
                )
            except BadRequest:
                try:
                    sent = await context.bot.edit_message_text(
                        chat_id=update.effective_chat.id,
                        message_id=edit,
                        text=chunk
                    )
                except BadRequest:
                    try:
                        if mark == '```':
                            mark = '\n```\n'
                        if sent.text != chunk[:4055]:
                            await context.bot.edit_message_text(
                                chat_id=update.effective_chat.id,
                                message_id=edit,
                                text=chunk[:4055]+mark,
                                parse_mode='Markdown'
                            )
                        chunk = mark+chunk[4055:]
                        mark = ""
                        sent = await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=chunk,
                            parse_mode='Markdown'
                        )
                        edit = sent.message_id
                    except BadRequest:
                        pass
            finally:
                last_time = time()
            #print(last_time)
    
    if sent is None:
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=edit,
                text=chunk,
                parse_mode='Markdown'
            )
        except BadRequest:
            pass
        finally:
            return wholeMessage

    if sent.text != chunk:
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=edit,
                text=chunk,
                parse_mode='Markdown'
            )
        except BadRequest:
            pass
        
    return wholeMessage


async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE, messages: list):
    file = context.user_data.get('file')
    command = ['sudo', 'systemctl', 'restart', 'ollama.service']
    sent = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text='.'
    )
    edit = sent.message_id
    async def generate_image(prompt: str, filename: str):
        
        await asyncio.create_subprocess_exec(*command)
        
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.jpg"
            
        res = tools.ratios.get(await DB.get_aspect_ratio(update.effective_chat.id))
        if res is None:
            res = (1440, 1440)
        

        model_id = "black-forest-labs/FLUX.1-schnell"
        torch.backends.cuda.matmul.allow_tf32 = True
        
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        pipe.enable_sequential_cpu_offload()

        seed = 0
        def _generate_image():
            image = pipe(
                prompt=prompt,
                output_type="pil",
                num_inference_steps=5,
                max_sequence_length=256,
                
                width=res[0],
                height=res[1],
                
                generator=torch.Generator().manual_seed(seed),
            ).images[0]
            image.save(f"{filename}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _generate_image)
        with open(filename, 'rb') as image:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=image
            )
        with open(filename, 'rb') as file:
            messages.append(message('tool', f"{filename} seed: {seed}"))
    
        if context.user_data.get('generate'):
            context.user_data.pop('generate')
    
        await context.bot.delete_message(
            chat_id=update.effective_chat.id,
            message_id=edit
        )
        
        if os.path.exists(filename):
            os.remove(filename)
    
    async def analyze_image(prompt: str, edit=edit):
        await asyncio.create_subprocess_exec(*command)
        await asyncio.sleep(1)
        multimodal = await AsyncClient().generate(model='llava', prompt=f"Prompt: {prompt}", images=[file.get('content')])
        translate = {'role': 'tool', 'content': f"Translate the text to the language of the prompt. Provide only the translation, nothing more. If the prompt is in English, don't translate, just copy the text. Text: {multimodal['response']} Prompt: {prompt}"}
        msgs = messages.copy()
        msgs.append(translate)
        
        wholeMessage = await write(update, context, msgs, edit)
        
        messages.append(message('tool', wholeMessage))
    async def analyze_document(prompt: str, edit=edit):
        await asyncio.create_subprocess_exec(*command)
        filecontent = await parser.extract(file)
        multimodal = await AsyncClient().generate(model='llava', prompt=f"Prompt: {prompt} . Text of the document: {filecontent[1]}", images=[filecontent[0]])
        translate = {'role': 'tool', 'content': f"Translate the text to the language of the prompt. Provide only the translation, nothing more. If the prompt is in English, don't translate, just copy the text. Text: {multimodal['response']} Prompt: {prompt}"}
        msgs = messages.copy()
        msgs.append(translate)
        
        wholeMessage = await write(update, context, msgs, edit)
        
        messages.append(message('tool', wholeMessage))
    
    
    if file:
        gen = await AsyncClient().chat(model=config['MODEL'], messages=messages, tools=tools.tools_in)
    else:
        if context.user_data.get('generate') == 'img':
            await generate_image(messages[-1]['content'][messages[-1]['content'].find(':')+2:], f"{update.effective_user.id}_{int(time())}")
            return
        else:
            gen = await AsyncClient().chat(model=config['MODEL'], messages=messages)
    
    gen_message = gen['message']
    
    #print(gen_message)
    if gen_message.get('tool_calls'):
        availableFunctions = {
            'generate_image': generate_image,
            'analyze_image': analyze_image,
            'analyze_document': analyze_document
        }
        for tool in gen_message['tool_calls']:
            toCall = availableFunctions[tool['function']['name']]
            args = tool['function']['arguments'].values()
            funcResp = await toCall(*args)
            #print('\n\n******')
            #print(funcResp)
    else:
        
        wholeMessage = await write(update, context, messages, edit)
        
        messages.append(message('assistant', wholeMessage))

    



@command_handler()
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('sent'):
        try:
            if context.user_data['sent'] != -1:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
            context.user_data.pop('sent')
        except:
            return
    if context.user_data.get('proc') and not context.user_data.get('generate'):
        return
    context.user_data['proc'] = True
    lang = context.user_data.get('lang')
    username = update.effective_user.full_name
    user = update.effective_user.id
    req = update.message.text if not context.args else ' '.join(context.args)
    msg = message('user', f"{user}, {username}: {req}")
    isNew = False
    chat_id = await DB.get_current_chat(update.effective_chat.id)
    if chat_id is None or chat_id == -1:
        isNew = True
        call = await DB.get_histories(update.effective_chat.id)
        chat_id = call[-1][0]+1 if call else 1
        
    path = f"chats/{update.effective_chat.id}/{chat_id}.json"
    
    directory = "/".join(path.split("/")[:-1])

    if not os.path.exists(directory):
        #isNew = True
        os.makedirs(directory)
    sent_status = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=M_LANG[lang]['generating'],
        parse_mode='Markdown'
    )
    edit_status = sent_status.message_id
    
    history = []
    if isNew is False:
        try:
            with open(path, mode='r', encoding='utf-8') as file:
                try:
                    history = (json.load(file)).get('content')
                except json.JSONDecodeError:
                    history = []
        except FileNotFoundError:
            history = []
            isNew = True
    messages = [SYSTEM]
    
    for i in history:
        messages.append(i)
    messages.append(msg)
    history.append(msg)
    
    
    await generate(update, context, messages)
    
    history = messages[1:]
    
    title = ""
    if isNew:
        title = await AsyncClient().chat(model=config['MODEL'], messages=[message('user', f"{config['GENTITLE']} {req}")])
        title = title['message']['content']
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=edit_status,
            text=f"{M_LANG[lang]['chat']}: {title}",
            parse_mode='Markdown'
        )
        await DB.set_current_chat(update.effective_chat.id, chat_id)
        await DB.set_history(chat_id, update.effective_chat.id, path)
    else:
        with open(path, mode='r', encoding='utf-8') as file:
            title = json.load(file).get('title')
        await context.bot.delete_message(
            chat_id=update.effective_chat.id,
            message_id=edit_status
        )
    
    if context.user_data.get('file'):
        context.user_data.pop('file')
    with open(path, mode='w+', encoding='utf-8') as file:
        file.write( json.dumps({"title": title, "content": history}, indent=4, ensure_ascii=False) )

    if context.user_data.get('proc'):
        context.user_data.pop('proc')

    return CHAT

@command_handler()
async def newchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await DB.set_current_chat(update.effective_chat.id, -1)
    
    return await chat(update, context)

@command_handler()
async def generateimage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['generate'] = 'img'
    
    return await chat(update, context)

@command_handler()
async def deletechat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('sent'):
        try:
            if context.user_data['sent'] != -1:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
            context.user_data.pop('sent')
        except:
            return
    else:
        return
    
    lang = context.user_data.get('lang')
    user = update.effective_user.id
    req = update.message.text if not context.args else ' '.join(context.args)
    
    try:
        req = int(req)
    except ValueError:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['invalid_type']
        )
        return ConversationHandler.END
    
    hist = await DB.get_history(req, update.effective_chat.id)
    
    if hist:
        with open(hist[1], mode='r', encoding='utf-8') as file:
            title = f"{hist[0]}) {json.load(file).get('title')}"
        await DB.delete_history(req, update.effective_chat.id)
        path = f"chats/{update.effective_chat.id}/{req}.json"
        hists = await DB.get_histories(update.effective_chat.id)
        if os.path.exists(hist[1]):
            os.remove(hist[1])
        if hists:
            for i in range(req-1, len(hists)):
                if os.path.exists(f"chats/{update.effective_chat.id}/{i+2}.json"):
                    os.rename(f"chats/{update.effective_chat.id}/{i+2}.json", f"chats/{update.effective_chat.id}/{i+1}.json")
                    await DB.delete_history(i+2, update.effective_chat.id)
                    await DB.set_history(i+1, update.effective_chat.id, f"chats/{update.effective_chat.id}/{i+1}.json")
            
        if await DB.get_current_chat(update.effective_chat.id) == req:
            await DB.set_current_chat(update.effective_chat.id, -1)
        
            
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"{M_LANG[lang]['chat_deleted']}:\n{title}"
        )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['not_found']
        )
    
    return ConversationHandler.END

@command_handler()
async def listchats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang')

    hist = await DB.get_histories(update.effective_chat.id)
    lst = []
    if hist is None:
        lst = ["--------"]
    else:    
        for i in hist:
            with open(i[1], mode='r', encoding='utf-8') as file:
                lst.append(f"{i[0]}) {json.load(file).get('title')}")
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"{M_LANG[lang]['chatlist']}:\n\n{'\n'.join(lst)}"
    )
    
    return ConversationHandler.END

@command_handler()
async def chattitle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('sent'):
        try:
            if context.user_data['sent'] != -1:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
            context.user_data.pop('sent')
        except:
            return
    else:
        return
    
    req = update.message.text if not context.args else ' '.join(context.args)
    lang = context.user_data.get('lang')
    
    try:
        req = int(req)
    except ValueError:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['invalid_type']
        )
        return ConversationHandler.END
    
    hist = await DB.get_history(req, update.effective_chat.id)
    
    if hist is None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['not_found']
        )
        return
    
    with open(hist[1], mode='r', encoding='utf-8') as file:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"{hist[0]}) {json.load(file).get('title')}"
        )
    
    return ConversationHandler.END
        
@command_handler(config['DELAY'])
async def currentchattitle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang')
    chat_id = await DB.get_current_chat(update.effective_chat.id)
    if chat_id is None or chat_id == -1:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['none_selected']
        )
        return
    
    hist = await DB.get_history(chat_id, update.effective_chat.id)
    
    if hist is None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['not_found']
        )
        return
    
    with open(hist[1], mode='r', encoding='utf-8') as file:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"{hist[0]}) {json.load(file).get('title')}"
        )
    

@command_handler()
async def chathistory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('sent'):
        try:
            if context.user_data['sent'] != -1:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
            context.user_data.pop('sent')
        except:
            return
    else:
        return
    
    req = update.message.text if not context.args else ' '.join(context.args)
    lang = context.user_data.get('lang')
    
    try:
        req = int(req)
    except ValueError:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['invalid_type']
        )
        return ConversationHandler.END
    
    hist = await DB.get_history(req, update.effective_chat.id)
    
    if hist is None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['not_found']
        )
        return
    
    with open(hist[1], mode='r', encoding='utf-8') as file:
        jsn = json.load(file).get('content')
        charCount = 0
        chunk = []
        for msg in jsn:
            index = msg['content']
            if msg['role'] == 'user':
                index = msg['content'][msg['content'].find(' ')+1:]
                
            if len(index) >= 3596:
                for i in range(0, len(index), 3596):
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=index[i:i+3596]
                    )
                continue
                
            if charCount + len(index) >= 3595:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text='\n\n'.join(chunk),
                    parse_mode='Markdown'
                )
                chunk = []
                charCount = 0
            chunk.append(index)
            charCount += len(index)
        
        if chunk:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text='\n\n'.join(chunk),
                parse_mode='Markdown'
            )
    
    return ConversationHandler.END
        
@command_handler(config['DELAY'])
async def currentchathistory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang')
    chat_id = await DB.get_current_chat(update.effective_chat.id)
    if chat_id is None or chat_id == -1:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['none_selected']
        )
        return
    
    hist = await DB.get_history(chat_id, update.effective_chat.id)
    with open(hist[1], mode='r', encoding='utf-8') as file:
        jsn = json.load(file).get('content')
        charCount = 0
        chunk = []
        for msg in jsn:
            index = msg['content']
            if msg['role'] == 'user':
                index = msg['content'][msg['content'].find(' ')+1:]
                
            if len(index) >= 4096:
                for i in range(0, len(index), 4096):
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=index[i:i+4096]
                    )
                continue
                
            if charCount + len(index) >= 4095:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text='\n\n'.join(chunk),
                    parse_mode='Markdown'
                )
                chunk = []
                charCount = 0
            chunk.append(index)
            charCount += len(index)
        
        if chunk:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text='\n\n'.join(chunk),
                parse_mode='Markdown'
            )

@command_handler()
async def selectchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('sent'):
        try:
            if context.user_data['sent'] != -1:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
            context.user_data.pop('sent')
        except:
            return
    else:
        return
    
    req = update.message.text if not context.args else ' '.join(context.args)
    lang = context.user_data.get('lang')
    
    try:
        req = int(req)
    except ValueError:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['invalid_type']
        )
        return ConversationHandler.END
    
    hist = await DB.get_history(req, update.effective_chat.id)
    
    if hist is None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['not_found']
        )
        return
    
    with open(hist[1], mode='r', encoding='utf-8') as file:
        name = json.load(file).get('title')
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"{M_LANG[lang]['select_chat']}: {name}"
        )
    
    await DB.set_current_chat(update.effective_chat.id, req)
    
    return ConversationHandler.END

@command_handler()
async def aspectratio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('sent'):
        try:
            if context.user_data['sent'] != -1:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
            context.user_data.pop('sent')
        except:
            return
    else:
        return
    
    req = update.message.text if not context.args else ' '.join(context.args)
    lang = context.user_data.get('lang')
    
    if req not in tools.ratios:
        try:
            req = ratios[int(req)]
        except (ValueError, KeyError):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=M_LANG[lang]['invalid_type']
            )
            return ConversationHandler.END
    
    await DB.set_aspect_ratio(update.effective_chat.id, req)
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"{M_LANG[lang]['ratio_set']}: {req}"
    )
    
    return ConversationHandler.END
        

@command_handler(config['DELAY'])
async def start_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await chat(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(M_LANG[lang]['query_request'], reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return CHAT

@command_handler(config['DELAY'])
async def start_newchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await newchat(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(M_LANG[lang]['query_request'], reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return NEWCHAT
    
@command_handler(config['DELAY'])
async def start_generateimage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await generateimage(update, context)
    else:
        lang = context.user_data.get('lang')
        ratio = await DB.get_aspect_ratio(update.effective_chat.id)
        if ratio is None:
            ratio = "1:1"
        sent = await update.message.reply_text(f"{M_LANG[lang]['query_prompt_request']}: ({ratio})", reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return GENERATEIMAGE

@command_handler(config['DELAY'])
async def start_deletechat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await deletechat(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(M_LANG[lang]['query_int_request'], reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return DELETECHAT

@command_handler(config['DELAY'])
async def start_chattitle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await chattitle(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(M_LANG[lang]['query_int_request'], reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return CHATTITLE

@command_handler(config['DELAY'])
async def start_chathistory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await chathistory(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(M_LANG[lang]['query_int_request'], reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return CHATHISTORY

@command_handler(config['DELAY'])
async def start_selectchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await selectchat(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(M_LANG[lang]['query_int_request'], reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return SELECTCHAT
    
@command_handler(config['DELAY'])
async def start_filechat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang')
    sent = await update.message.reply_text(M_LANG[lang]['query_file_request'], reply_markup=ForceReply(True))
    context.user_data['sent'] = sent.message_id
    return FILECHAT

@command_handler(config['DELAY'])
async def start_selectchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await selectchat(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(M_LANG[lang]['query_int_request'], reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return SELECTCHAT

@command_handler(config['DELAY'])
async def start_aspectratio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        context.user_data['sent'] = -1
        return await aspectratio(update, context)
    else:
        lang = context.user_data.get('lang')
        sent = await update.message.reply_text(f"{M_LANG[lang]['query_ratio_request']}\n{'\n'.join(list(map(lambda x: f"{x[0]}) {x[1]}", ratios.items())))}", reply_markup=ForceReply(True))
        context.user_data['sent'] = sent.message_id
        return ASPECTRATIO
    
    
@command_handler(config['DELAY'])
async def filechat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #file = update.message.effective_attachment[-1]
    if context.user_data.get('sent'):
        try:
            if context.user_data['sent'] != -1:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
            context.user_data.pop('sent')
        except:
            return
    
    if update.message.document:
        attach = update.message.document.file_id
        attach_name = update.message.document.file_name
    elif update.message.photo:
        attach = update.message.photo[-1].file_id
        attach_name = f"{update.effective_user.id}.jpg"
    else:
        lang = context.user_data.get('lang')
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=M_LANG[lang]['invalid_type']
        )
        return
    
    content = await context.bot.get_file(attach)
    byte = await content.download_as_bytearray()
    
    #print(attach_name, byte)
    context.user_data['file'] = {'name': attach_name, 'content': bytes(byte)}
    return await start_chat(update, context)



@command_handler()
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang')
    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
    context.user_data.pop('sent')
    await update.message.reply_text(M_LANG[lang]['query_cancel'])
    if not context.user_data.get('sent'):
        return ConversationHandler.END
    
@command_handler()
async def exit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang')
    if context.user_data.get('sent'):
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data['sent'])
        context.user_data.pop('sent')
    await update.message.reply_text(M_LANG[lang]['query_exit'])
    if not context.user_data.get('sent'):
        return ConversationHandler.END
    
async def init():
    global HAS_INIT
    if not HAS_INIT:
        await DB.connect()
        print('-'*20, "Bot Initialized", '-'*20, sep='\n')
        HAS_INIT = True

async def stop(application: Application):
    print('-'*20, "Shutting down", '-'*20, sep='\n')
    await DB.close()

if __name__ == '__main__':
    application = ApplicationBuilder().token(config['TOKEN']).post_shutdown(stop).write_timeout(90).read_timeout(90).concurrent_updates(True).build()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init())
    
    main_handler = ConversationHandler(
        entry_points=[CommandHandler('chat', start_chat),
                      CommandHandler('newchat', start_newchat),
                      CommandHandler('generateimage', start_generateimage),
                      CommandHandler('filechat', start_filechat),
                      CommandHandler('deletechat', start_deletechat),
                      CommandHandler('chattitle', start_chattitle),
                      CommandHandler('chathistory', start_chathistory),
                      CommandHandler('selectchat', start_selectchat),
                      CommandHandler('aspectratio', start_aspectratio)],
        states={
            CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, chat)],
            NEWCHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, newchat)],
            GENERATEIMAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, generateimage)],
            FILECHAT: [MessageHandler(filters.ATTACHMENT & ~filters.COMMAND & ~filters.TEXT, filechat)],
            DELETECHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, deletechat)],
            CHATTITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, chattitle)],
            CHATHISTORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, chathistory)],
            SELECTCHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, selectchat)],
            ASPECTRATIO: [MessageHandler(filters.TEXT & ~filters.COMMAND, aspectratio)]
        },
        fallbacks=[CommandHandler('exit', exit),
                   CommandHandler('cancel', cancel),
                   CommandHandler('chat', start_chat),
                   CommandHandler('newchat', start_newchat),
                   CommandHandler('generateimage', start_generateimage),
                   CommandHandler('filechat', start_filechat),
                   CommandHandler('deletechat', start_deletechat),
                   CommandHandler('chattitle', start_chattitle),
                   CommandHandler('chathistory', start_chathistory),
                   CommandHandler('selectchat', start_selectchat),
                   CommandHandler('aspectratio', start_aspectratio),
                   CommandHandler('start', start),
                   CommandHandler('language', language),
                   CommandHandler('listchats', listchats),
                   CommandHandler('currentchattitle', currentchattitle),
                   CommandHandler('currentchathistory', currentchathistory)],
    )
    
    start_handler = CommandHandler('start', start)
    language_handler = CommandHandler('language', language)
    listchats_handler = CommandHandler('listchats', listchats)
    currentchattitle_handler = CommandHandler('currentchattitle', currentchattitle)
    currentchathistory_handler = CommandHandler('currentchathistory', currentchathistory)
    
    application.add_handler(start_handler)
    application.add_handler(main_handler)
    application.add_handler(language_handler)
    application.add_handler(listchats_handler)
    application.add_handler(currentchattitle_handler)
    application.add_handler(currentchathistory_handler)
    application.add_handler(CallbackQueryHandler(button_click))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    application.add_handler(MessageHandler(filters.ATTACHMENT & ~filters.COMMAND, filechat))
    
    application.run_polling()
