#include <dl.h>
#include <assert.h>
#include <stdio.h>

dl_t dl_alloc()
{
  dl_t dl = (dl_t)malloc(sizeof(dl_s));
  if (!dl) {
      printf("Out of memory!!\n");
  } else {
    dl->first = dl->last = 0; dl->count = 0;
  }
  return dl;
}

void dl_delete(dl_t dl, dl_el *el)
{
  if (dl->first == el) {
    dl->first = el->next;
  }
  if (dl->last == el) {
    dl->last = el->prev;
  }
  if (el->next) {
    el->next->prev = el->prev;
  }
  if (el->prev) {
    el->prev->next = el->next;
  }
  free(el); dl->count--;
}

void dl_clear(dl_t dl)
{
  dl_el *el, *next;
  if (dl->count > 0) {
    for (el=dl->first; el; el=next) {
      next = el->next;
      free(el);
    }
  }
  dl->first = dl->last = 0;
  dl->count = 0;
}

void dl_concat(dl_t first_list, dl_t second_list)
{
  if (first_list->count <= 0) {
    *first_list = *second_list;
  } else if (second_list->count > 0) {
    first_list->last->next = second_list->first;
    second_list->first->prev = first_list->last;
    first_list->last = second_list->last;
    first_list->count += second_list->count;
  }

  free(second_list);
}

static void dl_insertion_sort(dl_t dl, size_t el_size,
			      int(*compar)(void *, void *))
{
  char *buf;
  void *curr_d, *srch_d;
  dl_el *curr, *srch;

  if (dl_length(dl) <= 1) {
    return;
  }

  buf = (char*)malloc(el_size);

  for (curr=dl->first; curr!=dl->last; curr=curr->next) {
    curr_d = (void*)(((dl_el*)curr)+1);

    for (srch=dl->last; srch!=curr; srch=srch->prev) {
      srch_d = (void*)(((dl_el*)srch)+1);
      if (compar(curr_d, srch_d) > 0) {
	memcpy((void*)buf, curr_d, el_size);	
	memcpy(curr_d, srch_d, el_size);
	memcpy(srch_d, (void*)buf, el_size);
      }
    }
  }
  

  free(buf);
}

void dl_sort(dl_t dl, size_t el_size, int(*compar)(void *, void *))
{
  dl_el *el, *first_head, *second_head;
  dl_s first_list, second_list;
  void *first_item, *second_item;
  int i, len;

  if (dl_length(dl) <= 25) {
    dl_insertion_sort(dl, el_size, compar);
    return;
  }

  len = dl_length(dl)/2;
  for (i=0, el=dl->first; i<len; i++) {
    el = el->next;
  }

  first_list.first = dl->first;
  first_list.last = el->prev;
  first_list.count = len;
  first_list.last->next = 0;

  second_list.first = el;
  second_list.last = dl->last;
  second_list.count = dl_length(dl)-len;
  second_list.first->prev = 0;

  dl_sort(&first_list, el_size, compar);
  dl_sort(&second_list, el_size, compar);

  /* in-place merging */
  first_head = first_list.first;
  second_head = second_list.first;

  first_item = (void*)(((dl_el*)first_head)+1);
  second_item = (void*)(((dl_el*)second_head)+1);
  if (compar(first_item, second_item) <= 0) {
    dl->first = el = first_head;
    first_head = first_head->next;
  } else {
    dl->first = el = second_head;
    second_head = second_head->next;
  }

  while (1) {
    first_item = (void*)(((dl_el*)first_head)+1);
    second_item = (void*)(((dl_el*)second_head)+1);
    if (compar(first_item, second_item) <= 0) {
      el->next = first_head;
      first_head->prev = el;
      el = first_head;
      first_head = first_head->next;
      if (!first_head) {
	el->next = second_head;
	second_head->prev = el;
	dl->last = second_list.last;
	break;
      }
    } else {
      el->next = second_head;
      second_head->prev = el;
      el = second_head;
      second_head = second_head->next;
      if (!second_head) {
	el->next = first_head;
	first_head->prev = el;
	dl->last = first_list.last;
	break;
      }
    }
  }
}
