def lambda_handler(event, context):
    """Lambda function handler"""
    try:
        # Load model
        interpreter = download_model()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Parse the input
        body = json.loads(event['body'])
        
        # Check if image is provided as base64
        if 'image' in body:
            image_data = base64.b64decode(body['image'])
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }
        
        # Preprocess the image
        input_data = preprocess_image(image_data)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Process results
        predictions = output_data[0].tolist()
        
        # Create response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'predictions': results,
                'top_emotion': results[0]['emotion'],
                'confidence': results[0]['confidence']
            })
        }
        
        return response
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }