# =============================================================================
#
# H2O.ai Wave Computer Vision App
#
# @Author =  Kostyantyn Stokolos
#
# Version 1.0.1 
#   -- Added:
#       - Upload Image Changed from backend to 
#
# =============================================================================

# =========================== IMPORTS =========================================
# UI, UX
from h2o_wave import Q, ui, main, app

# user scripts
from model import EncoderCNN, DecoderRNN
from caption_generator import setup_ml, generate_captions

print('Imports Succesfull')


# =========================== FUNCTION APP ====================================
def setup_app(q: Q, path: str) -> None:
    """
    Activities that happen the first time someone comes to this app, such as user variables and home page cards
    :param q: Query argument from the H2O Wave server
    :param path: path 
    :return: None
    """
    print("Setting up the app for a new browser tab..")

    # header 
    q.page['header'] = ui.form_card(
            box='1 1 6 5',
            items=[
                ui.text_xl('Image Uploader'),
                ui.file_upload(
                    name='upload_image',
                    label='Upload Image',
                    multiple=False
                ),       
            ])        
    
    # base image where the uploaded image will be
    q.page['image'] = ui.image_card(
        box='1 6 6 9',
        title='Your Image',
        type='jpg',
        path=path
    )

    # text where generated captions are going to be
    q.page['text'] = ui.form_card(
        box='1 15 6 2',
        items=[
            ui.text_xl('Upload an image and I will tell you what I see..')
        ],
    )
    q.client.initialized = True


def upload_image(q: Q, message: str) -> None:
    """
    Creating UI when the user wants to upload the image so automatic captions can
    be generated.
    :param q: Query argument from the H2O Wave server
    :return: None
    """
    
    # Uploader displaying the back button
    q.page['header'] = ui.form_card(
        box='1 1 6 2',
        items=[
            ui.text_xl('Image Uploader'),
            ui.button(
                name='show_form',
                label='Back', 
                primary=True
            ),
        ]
    )

    # uploaded image
    q.page['image'] = ui.image_card(
            box='1 3 6 8',
            title='Your Image',
            type='jpg',
            path=q.args.upload_image[0]
        )
        
    # show text generated by ML
    q.page['text'] = ui.form_card(
        box='1 11 6 2',
        items=[
            ui.text_xl(f'I see: {message}')
        ],
    )      
    
# =========================== LOGIC ===========================================
@app('/demo')
async def serve(q: Q) -> None:
    """
    Handle interactions from the browser such as new arrivals and button clicks
    :param q: Query argument from the H2O Wave server
    :return: None
    """
    # upload default image
    default_image_path, = await q.site.upload(['./static/default_image.jpeg'])   
    print("Handling a user event..")
    if not q.client.initialized:   
        # default setup (firts time appearance fo the app)
        setup_app(q, default_image_path)
    
    elif q.args.upload_image:
        
        # upload image and trigger ML component
        transformer, encoder, decoder, vocab = setup_ml()
        local_path = await q.site.download(q.args.upload_image[0], '.')
        message = await q.run(generate_captions, local_path, vocab, transformer, encoder, decoder)
        upload_image(q, message)
        print(f"-- Captions generated: {message}")
        
    elif q.args.show_form:      
        # default setup (firts time appearance fo the app)
        setup_app(q, default_image_path)
    
    await q.page.save()

