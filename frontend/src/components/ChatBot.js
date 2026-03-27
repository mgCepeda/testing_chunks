import React, { Component } from 'react';
import '../stylesheets/ChatBot.css';

class ChatBot extends Component {
  constructor(props) {
    super(props);
    this.state = {
      messages: [
        {
          role: 'bot',
          text: '¡Bienvenido! Soy tu experto literario. Pregúntame sobre Dune, El Señor de los Anillos, Harry Potter o Juego de Tronos.',
        },
      ],
      input: '',
      status: 'idle', // 'idle' | 'loading' | 'streaming'
    };
    this.messagesEndRef = React.createRef();
  }

  componentDidUpdate() {
    if (this.messagesEndRef.current) {
      this.messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }

  sendMessage = async () => {
    const question = this.state.input.trim();
    if (!question || this.state.status !== 'idle') return;

    const userMessage = { role: 'user', text: question };
    this.setState((prev) => ({
      messages: [...prev.messages, userMessage],
      input: '',
      status: 'loading',
    }));

    let botMessageAdded = false;

    try {
      const response = await fetch('http://localhost:5001/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter((l) => l.startsWith('data: '));

        for (const line of lines) {
          const data = JSON.parse(line.replace('data: ', ''));
          if (data.done) {
            this.setState({ status: 'idle' });
          } else if (data.token) {
            if (!botMessageAdded) {
              this.setState((prev) => ({
                messages: [...prev.messages, { role: 'bot', text: data.token }],
                status: 'streaming',
              }));
              botMessageAdded = true;
            } else {
              this.setState((prev) => {
                const messages = [...prev.messages];
                messages[messages.length - 1] = {
                  role: 'bot',
                  text: messages[messages.length - 1].text + data.token,
                };
                return { messages };
              });
            }
          }
        }
      }
      if (this.state.status !== 'idle') {
        this.setState({ status: 'idle' });
      }
    } catch (error) {
      this.setState((prev) => ({
        messages: [...prev.messages, {
          role: 'bot',
          text: 'Error al conectar con el servidor. Asegúrate de que la API está corriendo.',
        }],
        status: 'idle',
      }));
    }
  };

  handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.sendMessage();
    }
  };

  render() {
    return (
      <div id='chat-view'>
        <div className='chat-header'>
          <span className='chat-icon'>📚</span>
          <h2>Experto Literario</h2>
        </div>

        <div className='chat-messages'>
          {this.state.messages.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.role}`}>
              <span className='bubble-icon'>{msg.role === 'bot' ? '🎓' : '👤'}</span>
              <p>{msg.text}</p>
            </div>
          ))}
          {this.state.status === 'loading' && (
            <div className='chat-bubble bot'>
              <span className='bubble-icon'>🎓</span>
              <div>
                <p className='typing'>Pensando<span>.</span><span>.</span><span>.</span></p>
                <p className='wait-hint'>El modelo puede tardar hasta un minuto</p>
              </div>
            </div>
          )}
          <div ref={this.messagesEndRef} />
        </div>

        <div className='chat-input-area'>
          <textarea
            value={this.state.input}
            onChange={(e) => this.setState({ input: e.target.value })}
            onKeyDown={this.handleKeyDown}
            placeholder='Escribe tu pregunta literaria...'
            rows={2}
            disabled={this.state.status !== 'idle'}
          />
          <button
            onClick={this.sendMessage}
            disabled={this.state.status !== 'idle' || !this.state.input.trim()}
          >
            Enviar
          </button>
        </div>
      </div>
    );
  }
}

export default ChatBot;
