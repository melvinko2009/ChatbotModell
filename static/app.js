class Chatbox{
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
        }
        this.state = false;
        this.messages = []
    }

    display(){
        const{openButton,chatBox, sendButton} = this.args;
        openButton.addEventListener("click", () => this.toggleState(chatBox))
        sendButton.addEventListener("click", () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input')
        node.addEventListener('keyup',({key}) => {
            if (key === "Enter"){
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatBox){
        this.state = !this.state;
        if(this.state){
            chatBox.classList.add('chatbox--active')
        }else{
            chatBox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox){
        var textField = chatbox.querySelector('input')
        let text = textField.value
        if (text === "") return

        let msg = {name : "User", message : text}
        this.messages.push(msg)

        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({message:text}),
            mode: "cors",
            headers: { 'Content-Type': 'application/json'},
        })
            .then(r => r.json())
            .then(r => {
                let newMsg = {name: "MerckBot", message: r.answer};
                this.messages.push(newMsg);
                this.updateChatText(chatbox);
                textField.value = ""
            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox);
                textField.value = ""
        });
    }

    updateChatText(chatbox){
        let html = '';
        this.messages.slice().reverse().forEach(function (item,){
            if (item.name === "MerckBot"){
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }else{
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }

            const chatMessage = chatbox.querySelector('.chatbox__messages');
            chatMessage.innerHTML = html;
        })
    }
}

const chatBox = new Chatbox();
chatBox.display()