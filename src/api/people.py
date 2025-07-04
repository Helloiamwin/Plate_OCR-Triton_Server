from fastapi import APIRouter, Response, status
from typing import List
from urllib.parse import unquote
from src.app.person_crud import PersonCRUD

person_crud_instance = PersonCRUD()

router = APIRouter(prefix='/people', tags=['people'])

@router.get("",status_code=status.HTTP_200_OK)
async def get_all_people(skip: int = 0, limit: int = 10) -> list:
	person_list = person_crud_instance.select_all_people(skip, limit)
	return person_list

@router.get(
	"/{person_id}",
	status_code=status.HTTP_200_OK
)
async def select_person_by_ID(person_id: str):
	person_id = unquote(person_id)
	person_doc = person_crud_instance.select_person_by_id(person_id)
	return person_doc

@router.post(
	"",
	status_code=status.HTTP_201_CREATED	
)
async def insert_one_person(id: str, name: str):
	id,name = unquote(id), unquote(name)
	person_doc = person_crud_instance.insert_person(id, name)
	return person_doc

@router.post(
	"/{person_id}/name",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
async def update_name(person_id: str, name: str):
	person_id, name = unquote(person_id), unquote(name)
	person_crud_instance.update_person_name(person_id, name)

@router.delete(
	"/{id}",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT	
)
async def delete_person_by_ID(id: str):
	person_crud_instance.delete_person_by_id(id)


@router.delete(
	"",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
async def delete_all_people():
	person_crud_instance.delete_all_people()
