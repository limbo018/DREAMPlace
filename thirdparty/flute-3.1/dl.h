#ifndef DL_H
#define DL_H

#include <string.h>
#include <stdlib.h>

typedef struct dl_el_s {
  struct dl_el_s *prev, *next;
} dl_el;

typedef struct {
  dl_el *first, *last;
  unsigned int count;
} dl_s, *dl_t;

dl_t dl_alloc(void);
void dl_delete(dl_t dl, dl_el *el);
void dl_clear(dl_t dl);
void dl_concat(dl_t list1, dl_t list2);
void dl_sort(dl_t dl, size_t el_size, int(*compar)(void *, void *));

#define dl_length(dl) (dl)->count

#define dl_empty(dl) ((dl)->count <= 0)

#define dl_data(type, el) \
  *(type*)(((dl_el*)(el))+1)

#define dl_data_p(type, el) \
  ((type*)(((dl_el*)(el))+1))

#define dl_forall(type, dl, data) \
{ \
  dl_el *_el, *_next; \
  dl_t _curr_dl = (dl); \
  for (_el=_curr_dl->first; _el; _el=_next) { \
    _next = _el->next; \
    (data) = dl_data(type, _el);

#define dl_forall_p(type, dl, data_p) \
{ \
  dl_el *_el, *_next; \
  dl_t _curr_dl = (dl); \
  for (_el=_curr_dl->first; _el; _el=_next) { \
    _next = _el->next; \
    (data_p) = dl_data_p(type, _el);

#define dl_current() _el
#define dl_delete_current() dl_delete(_curr_dl, _el)

#define dl_endfor \
  } \
}

#define dl_forall_reverse(type, dl, data) \
{ \
  dl_el *_el, *_next; \
  dl_t _curr_dl = (dl); \
  for (_el=_curr_dl->last; _el; _el=_next) { \
    _next = _el->prev; \
    (data) = dl_data(type, _el);

#define dl_forall_reverse_p(type, dl, data_p) \
{ \
  dl_el *_el, *_next; \
  dl_t _curr_dl = (dl); \
  for (_el=_curr_dl->last; _el; _el=_next) { \
    _next = _el->prev; \
    (data_p) = dl_data_p(type, _el);

#define dl_first(type, dl) \
  dl_data(type, (dl)->first)

      
#define dl_first_element(dl) (dl)->first 

      
#define dl_last(type, dl) \
  dl_data(type, (dl)->last)

#define dl_pop_first(type, dl, data) \
{ \
  (data) = dl_first(type, dl); \
  dl_delete((dl), (dl)->first); \
}

#define dl_pop_last(type, dl, data) \
{ (data) = dl_last(type, dl);  dl_delete((dl), (dl)->last); }

#define dl_insert_before(type, dl, element, data) \
{ \
  if ((element) == (dl)->first) { \
    dl_prepend(type, dl, data); \
  } else { \
    dl_el *_el = (dl_el*) malloc(sizeof(dl_el)+sizeof(type)); \
    if (!_el) { \
      printf("Out of memory!!\n"); \
    } else { \
      memcpy(_el+1, &(data), sizeof(type)); \
      _el->prev = (element)->prev; _el->next = (element); \
      (element)->prev->next = _el; (element)->prev = _el; \
      (dl)->count++; \
    } \
  } \
}

#define dl_insert_after(type, dl, element, data) \
{ \
  if ((element) == (dl)->last) { \
    dl_append(type, dl, data); \
  } else { \
    dl_el *_el = (dl_el*) malloc(sizeof(dl_el)+sizeof(type)); \
    if (!_el) { \
	printf("Out of memory!!\n"); \
    } else { \
      memcpy(_el+1, &(data), sizeof(type)); \
      _el->next = (element)->next; _el->prev = (element); \
      (element)->next->prev = _el; (element)->next = _el; \
      (dl)->count++; \
    } \
  } \
}

#define dl_append(type, dl, data) \
{ \
  dl_el *_el = (dl_el*) malloc(sizeof(dl_el)+sizeof(type)); \
  if (!_el) { \
      printf("Out of memory!!\n"); \
  } else { \
    memcpy(_el+1, &(data), sizeof(type)); \
    _el->next = 0; \
    if ((dl)->count <= 0) { \
      _el->prev = 0; \
      (dl)->first = (dl)->last = _el; \
      (dl)->count = 1; \
    } else { \
      _el->prev = (dl)->last; \
      (dl)->last->next = _el; \
      (dl)->last = _el; \
      (dl)->count++; \
    } \
  } \
}

#define dl_prepend(type, dl, data) \
{ \
  dl_el *_el = (dl_el*) malloc(sizeof(dl_el)+sizeof(type)); \
  if (!_el) { \
    printf("Out of memory!!\n"); \
  } else { \
    memcpy(_el+1, &(data), sizeof(type)); \
    _el->prev = 0; \
    if ((dl)->count <= 0) { \
      _el->next = 0; \
      (dl)->first = (dl)->last = _el; \
      (dl)->count = 1; \
    } else { \
      _el->next = (dl)->first; \
      (dl)->first->prev = _el; \
      (dl)->first = _el; \
      (dl)->count++; \
    } \
  } \
} 

#define dl_free(dl) \
{ \
  dl_clear(dl);  free(dl);  dl = 0; \
}

#define dl_duplicate(dest, src, type) \
{ \
  dest = dl_alloc(); \
  type _data_el; \
  dl_forall(type, src, _data_el) { \
    dl_append(type, dest, _data_el); \
  } dl_endfor; \
}

#endif
