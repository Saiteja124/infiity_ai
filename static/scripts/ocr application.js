
const ocrOption = document.querySelector('.ocr-option');
const chatbotOption = document.querySelector('.chatbot-option');
const ocrContent = document.querySelector('.ocr-content');
const chatbotContent = document.querySelector('.chatbot-content');

ocrOption.addEventListener('click', function() {
    ocrContent.classList.add('active');
    chatbotContent.classList.remove('active');
});

chatbotOption.addEventListener('click', function() {
    chatbotContent.classList.add('active');
    ocrContent.classList.remove('active');
});


    //const ocrOption = document.querySelector('.ocr-option');
    //const chatbotOption = document.querySelector('.chatbot-option');
    //const ocrContent = document.querySelector('.ocr-content');
   // const chatbotContent = document.querySelector('.chatbot-content');

    //ocrOption.addEventListener('click', function() {
      //  ocrContent.classList.add('active');
        //chatbotContent.classList.remove('active');
    //});

    //chatbotOption.addEventListener('click', function() {
      //  chatbotContent.classList.add('active');
        //ocrContent.classList.remove('active');
    //});
