from app.schemas.output import PipelineResponse


class Pipeline:
    def __init__(self, vision_service, rag_service, reasoning_service):
        self.vision_service = vision_service
        self.rag_service = rag_service
        self.reasoning_service = reasoning_service

    def run(self, image_path: str, empresa: str, setor: str) -> PipelineResponse:
        detected_people = self.vision_service.process(image_path)
        rules = self.rag_service.retrieve(empresa, setor)

        results = []
        for person in detected_people:
            person_result = self.reasoning_service.analyze(person, rules)
            results.append(person_result)

        return PipelineResponse(results=results)
